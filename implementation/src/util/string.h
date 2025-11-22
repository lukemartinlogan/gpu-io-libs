#pragma once

#include "../serialization/serialization.h"
#include "../hdf5/types.h"

inline hdf5::string ReadNullTerminatedString(Deserializer& de, cstd::optional<size_t> max_size) {
    hdf5::string str;

    while (!max_size || str.size() < *max_size) {
        auto c = static_cast<char>(de.Read<byte_t>());

        if (c == '\0') {
            break;
        }

        str.push_back(c);
    }

    return str;
}