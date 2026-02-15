#pragma once

#ifdef HDF5_CPU_BASELINE_ENABLED

#include <hdf5.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace hdf5_ref {

// RAII wrapper for HDF5 file handle
class File {
public:
    explicit File(const std::string& path, bool read_write = false) {
        unsigned flags = read_write ? H5F_ACC_RDWR : H5F_ACC_RDONLY;
        file_id_ = H5Fopen(path.c_str(), flags, H5P_DEFAULT);
        if (file_id_ < 0) {
            throw std::runtime_error("Failed to open HDF5 file: " + path);
        }
    }

    // Create a new file (truncates if exists)
    static File create(const std::string& path) {
        File f;
        f.file_id_ = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (f.file_id_ < 0) {
            throw std::runtime_error("Failed to create HDF5 file: " + path);
        }
        return f;
    }

    ~File() {
        if (file_id_ >= 0) {
            H5Fclose(file_id_);
        }
    }

    File(const File&) = delete;
    File& operator=(const File&) = delete;

    File(File&& other) noexcept : file_id_(other.file_id_) {
        other.file_id_ = -1;
    }
    File& operator=(File&& other) noexcept {
        if (this != &other) {
            if (file_id_ >= 0) H5Fclose(file_id_);
            file_id_ = other.file_id_;
            other.file_id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return file_id_; }

    // Create a group
    hid_t create_group(const std::string& name) {
        hid_t group_id = H5Gcreate2(file_id_, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group_id < 0) {
            throw std::runtime_error("Failed to create group: " + name);
        }
        return group_id;
    }

private:
    File() = default;
    hid_t file_id_ = -1;
};

// RAII wrapper for HDF5 dataset handle
class Dataset {
public:
    Dataset(hid_t file_id, const std::string& name) {
        dataset_id_ = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
        if (dataset_id_ < 0) {
            throw std::runtime_error("Failed to open dataset: " + name);
        }
        dataspace_id_ = H5Dget_space(dataset_id_);
    }

    // Create a new 1D dataset
    template<typename T>
    static Dataset create_1d(hid_t file_id, const std::string& name, hsize_t size) {
        Dataset ds;
        hsize_t dims[1] = {size};
        ds.dataspace_id_ = H5Screate_simple(1, dims, nullptr);
        ds.dataset_id_ = H5Dcreate2(file_id, name.c_str(), get_hdf5_type<T>(),
                                     ds.dataspace_id_, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (ds.dataset_id_ < 0) {
            H5Sclose(ds.dataspace_id_);
            throw std::runtime_error("Failed to create dataset: " + name);
        }
        return ds;
    }

    ~Dataset() {
        if (dataspace_id_ >= 0) H5Sclose(dataspace_id_);
        if (dataset_id_ >= 0) H5Dclose(dataset_id_);
    }

    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    Dataset(Dataset&& other) noexcept
        : dataset_id_(other.dataset_id_), dataspace_id_(other.dataspace_id_) {
        other.dataset_id_ = -1;
        other.dataspace_id_ = -1;
    }
    Dataset& operator=(Dataset&& other) noexcept {
        if (this != &other) {
            if (dataspace_id_ >= 0) H5Sclose(dataspace_id_);
            if (dataset_id_ >= 0) H5Dclose(dataset_id_);
            dataset_id_ = other.dataset_id_;
            dataspace_id_ = other.dataspace_id_;
            other.dataset_id_ = -1;
            other.dataspace_id_ = -1;
        }
        return *this;
    }

    [[nodiscard]] hid_t id() const { return dataset_id_; }
    [[nodiscard]] hid_t space() const { return dataspace_id_; }

    // Read a single element at the given flat index
    template<typename T>
    T read_element(hsize_t index) {
        T value;

        // Create memory dataspace for single element
        hsize_t mem_dims[1] = {1};
        hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);

        // Select the element in the file dataspace
        hsize_t start[1] = {index};
        hsize_t count[1] = {1};
        H5Sselect_hyperslab(dataspace_id_, H5S_SELECT_SET, start, nullptr, count, nullptr);

        // Read
        H5Dread(dataset_id_, get_hdf5_type<T>(), mem_space, dataspace_id_, H5P_DEFAULT, &value);

        H5Sclose(mem_space);
        return value;
    }

    // Read multiple elements starting at offset
    template<typename T>
    void read_sequential(T* buffer, hsize_t offset, hsize_t count) {
        hsize_t mem_dims[1] = {count};
        hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);

        hsize_t start[1] = {offset};
        hsize_t cnt[1] = {count};
        H5Sselect_hyperslab(dataspace_id_, H5S_SELECT_SET, start, nullptr, cnt, nullptr);

        H5Dread(dataset_id_, get_hdf5_type<T>(), mem_space, dataspace_id_, H5P_DEFAULT, buffer);

        H5Sclose(mem_space);
    }

    // Read elements at random indices
    template<typename T>
    void read_random(T* buffer, const hsize_t* indices, hsize_t count) {
        hsize_t mem_dims[1] = {count};
        hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);

        // Use point selection for random access
        H5Sselect_elements(dataspace_id_, H5S_SELECT_SET, count, indices);

        H5Dread(dataset_id_, get_hdf5_type<T>(), mem_space, dataspace_id_, H5P_DEFAULT, buffer);

        H5Sclose(mem_space);
    }

    // Write multiple elements starting at offset
    template<typename T>
    void write_sequential(const T* buffer, hsize_t offset, hsize_t count) {
        hsize_t mem_dims[1] = {count};
        hid_t mem_space = H5Screate_simple(1, mem_dims, nullptr);

        hsize_t start[1] = {offset};
        hsize_t cnt[1] = {count};
        H5Sselect_hyperslab(dataspace_id_, H5S_SELECT_SET, start, nullptr, cnt, nullptr);

        H5Dwrite(dataset_id_, get_hdf5_type<T>(), mem_space, dataspace_id_, H5P_DEFAULT, buffer);

        H5Sclose(mem_space);
    }

    template<typename T>
    static hid_t get_hdf5_type() {
        if constexpr (std::is_same_v<T, double>) {
            return H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, float>) {
            return H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return H5T_NATIVE_INT32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return H5T_NATIVE_INT64;
        } else {
            static_assert(sizeof(T) == 0, "Unsupported type for HDF5");
        }
    }

private:
    Dataset() = default;
    hid_t dataset_id_ = -1;
    hid_t dataspace_id_ = -1;
};

} // namespace hdf5_ref

#endif // HDF5_CPU_BASELINE_ENABLED
