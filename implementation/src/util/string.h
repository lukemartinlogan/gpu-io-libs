#pragma once

#include "../serialization/serialization.h"
#include "../hdf5/types.h"

inline hdf5::expected<hdf5::string> ReadNullTerminatedString(VirtualDeserializer& de, size_t max_size = hdf5::gpu_string<>::max_size()) {
    hdf5::string str;
    bool found = false;

    ASSERT(max_size <= hdf5::gpu_string<>::max_size(), "max_size must be less than inplace string max size");

    while (str.size() < max_size) {
        auto c = static_cast<char>(de.Read<byte_t>());

        if (c == '\0') {
            found = true;
            break;
        }

        str.push_back(c);
    }

    if (!found) {
        return hdf5::error(hdf5::HDF5ErrorCode::StringNotNullTerminated, "string read was not null-terminated");
    }

    return str;
}