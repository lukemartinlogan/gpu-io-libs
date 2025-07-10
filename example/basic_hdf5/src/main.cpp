#include <H5Cpp.h>
#include <hdf5.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include "hwinfo.h"
#include "sample.h"

struct SamplerParams {
    std::filesystem::path file_path;

    std::string group_name;
    uint64_t sample_ms;
    uint16_t sample_ct;

    enum class Api { C, CPP } api;
};

std::vector<Sample> collect_samples(uint64_t sample_ms, uint16_t sample_ct) {
    std::vector<Sample> samples;
    samples.reserve(sample_ct);

    for (size_t s = 0; s < sample_ct; ++s) {
        samples.push_back({
            .timestamp_ms = current_time_ms(),
            .memory_kb = mem_kb(),
            .cpu_time = cpu_time(),
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
    }

    return samples;
}

int hdf5_cpp(const SamplerParams& params) {
    try {
        unsigned int flags = std::filesystem::exists(params.file_path) ? H5F_ACC_RDWR : H5F_ACC_TRUNC;

        H5::H5File file(params.file_path, flags);

        H5::Group run_group = file.createGroup(params.group_name);

        hsize_t dims[1] = { params.sample_ct };
        H5::DataSpace dataspace(1, dims);

        H5::CompType data_type = Sample::h5cpp_comp_ty();

        H5::DataSet dataset = run_group.createDataSet("samples", data_type, dataspace);

        std::vector samples = collect_samples(params.sample_ms, params.sample_ct);

        dataset.write(samples.data(), data_type);

        H5::StrType str_ty(H5::PredType::C_S1, H5T_VARIABLE);
        H5::DataSpace scalar_space(H5S_SCALAR);

        H5::Attribute unit_attr = run_group.createAttribute("units", str_ty, scalar_space);
        unit_attr.write(str_ty, "timestamp: ms, cpu: %, memory: kB");

        run_group.createAttribute("sampling_interval_ms", H5::PredType::NATIVE_UINT64, scalar_space)
            .write(H5::PredType::NATIVE_UINT64, &params.sample_ms);

        run_group.createAttribute("sample_count", H5::PredType::NATIVE_UINT16, scalar_space)
            .write(H5::PredType::NATIVE_UINT16, &params.sample_ct);

        std::cout << std::format("wrote data to {} in {}.", dataset.getObjName(), params.file_path.string()) << std::endl;

        return 0;
    } catch (H5::Exception& e) {
        std::cout << "HDF5 error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "unknown error" << std::endl;
        return 1;
    }
}

int hdf5_c(const SamplerParams& params) {
    try {
        hid_t file;

        // 1. open / create the file
        if (std::filesystem::exists(params.file_path)) {
            file = H5Fopen(params.file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        } else {
            file = H5Fcreate(params.file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        }

        if (file < 0) {
            throw std::runtime_error("Failed to open / create file");
        }

        // 2. create the run group
        hid_t run_group = H5Gcreate(file, params.group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (run_group < 0) {
            throw std::runtime_error("Failed to create group");
        }

        // 3. dataspace
        hsize_t dims[1] = { params.sample_ct };

        hid_t dataspace = H5Screate_simple(1, dims, dims);

        // 4. compound type
        hid_t data_type = Sample::h5_comp_ty();

        // 5. create dataset
        hid_t dataset = H5Dcreate2(run_group, "samples", data_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // 6. collect & write samples
        std::vector samples = collect_samples(params.sample_ms, params.sample_ct);

        H5Dwrite(dataset, data_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, samples.data());

        // 7. write attributes
        hid_t str_ty = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_ty, H5T_VARIABLE);

        hid_t scalar_space = H5Screate(H5S_SCALAR);
        hid_t units_attr = H5Acreate(run_group, "units", str_ty, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
        const char* units = "timestamp: ms, cpu: %, memory: kB";
        H5Awrite(units_attr, str_ty, &units);

        hid_t interval_attr = H5Acreate(run_group, "sampling_interval_ms", H5T_NATIVE_UINT64, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(interval_attr, H5T_NATIVE_UINT64, &params.sample_ms);

        hid_t sample_ct_attr = H5Acreate(run_group, "sample_count", H5T_NATIVE_UINT16, scalar_space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(sample_ct_attr, H5T_NATIVE_UINT16, &params.sample_ct);

        std::cout << std::format("wrote data to samples in {}.", params.file_path.string()) << std::endl;

        // 8. close in reverse order of create
        H5Aclose(sample_ct_attr);
        H5Aclose(interval_attr);
        H5Aclose(units_attr);
        H5Sclose(scalar_space);
        H5Tclose(str_ty);
        H5Dclose(dataset);
        H5Tclose(data_type);
        H5Sclose(dataspace);
        H5Gclose(run_group);
        H5Fclose(file);

        return 0;
    } catch (std::exception& e) {
        std::cout << "HDF5 error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "unknown error" << std::endl;
        return 1;
    }
}

int main() {
    SamplerParams params{};

    params.file_path = "./system_info.h5";
    params.api = SamplerParams::Api::C;

    std::cout << "Group name: ";
    std::getline(std::cin, params.group_name);

    std::cout << "Sample count: ";
    if (!(std::cin >> params.sample_ct)) {
        throw std::runtime_error("given sample count was not integer");
    }

    std::cout << "Sample interval (ms): ";
    if (!(std::cin >> params.sample_ms)) {
        throw std::runtime_error("given sample interval was not integer");
    }

    if (params.api == SamplerParams::Api::C) { // NOLINT
        return hdf5_c(params);
    } else {
        return hdf5_cpp(params);
    }
}
