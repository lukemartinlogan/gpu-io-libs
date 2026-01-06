// TODO(kernel-hang): TEMPORARILY STUBBED to fix GPU kernel hang

#include "dataset.h"

__device__
hdf5::expected<Dataset> Dataset::New(const Object& object) {
    return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "Dataset::New temporarily disabled");
}

__device__
hdf5::expected<void> Dataset::Read(cstd::span<byte_t> buffer, size_t start_index, size_t count) const {
    return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "Dataset::Read temporarily disabled");
}

__device__
hdf5::expected<void> Dataset::Write(cstd::span<const byte_t> data, size_t start_index) const {
    return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "Dataset::Write temporarily disabled");
}

__device__
hdf5::expected<void> Dataset::ReadHyperslab(
    cstd::span<byte_t> buffer,
    const hdf5::dim_vector<uint64_t>& start,
    const hdf5::dim_vector<uint64_t>& count,
    const hdf5::dim_vector<uint64_t>& stride,
    const hdf5::dim_vector<uint64_t>& block
) const {
    return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "Dataset::ReadHyperslab temporarily disabled");
}

__device__
hdf5::expected<void> Dataset::WriteHyperslab(
    cstd::span<const byte_t> data,
    const hdf5::dim_vector<uint64_t>& start,
    const hdf5::dim_vector<uint64_t>& count,
    const hdf5::dim_vector<uint64_t>& stride,
    const hdf5::dim_vector<uint64_t>& block
) const {
    return hdf5::error(hdf5::HDF5ErrorCode::NotImplemented, "Dataset::WriteHyperslab temporarily disabled");
}
