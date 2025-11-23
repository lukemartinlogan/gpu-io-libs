#pragma once

#include "../serialization/serialization.h"
#include "../hdf5/types.h"

template<serde::Deserializer D>
hdf5::expected<hdf5::string> ReadNullTerminatedString(D& de, size_t max_size = hdf5::gpu_string<>::max_size()) {
    hdf5::string str;
    bool found = false;

    ASSERT(max_size <= hdf5::gpu_string<>::max_size(), "max_size must be less than inplace string max size");

    while (str.size() < max_size) {
        auto c = static_cast<char>(serde::Read<byte_t>(de));

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

// TODO: does this forget about the null byte sometimes?
template<serde::Serializer S>
void WritePaddedString(hdf5::string_view name, S& s) {
    size_t name_size = name.size();

    // write string
    s.WriteBuffer(cstd::span(
        reinterpret_cast<const byte_t*>(name.data()),
        name_size
    ));

    // pad to 8 bytes
    size_t padding = (name_size / 8 + 1) * 8 - name_size;
    static constexpr cstd::array<byte_t, 8> nul_bytes{};

    s.WriteBuffer(cstd::span(nul_bytes.data(), padding));
}

template<serde::Deserializer D>
hdf5::expected<hdf5::string> ReadPaddedString(D& de) {
    hdf5::string name;

    for (;;) {
        // 8 byte blocks
        cstd::array<byte_t, 8> buf{};

        if (!de.ReadBuffer(buf)) {
            return hdf5::error(hdf5::HDF5ErrorCode::BufferTooSmall, "failed to read string block");
        }

        auto nul_pos = std::ranges::find(buf, static_cast<byte_t>('\0'));

        auto append_result = name.append(hdf5::string_view(
            reinterpret_cast<const char*>(buf.data()),
            std::distance(buf.begin(), nul_pos)
        ));

        if (!append_result) {
            return cstd::unexpected(append_result.error());
        }

        if (nul_pos != buf.end()) {
            break;
        }
    }

    return name;
}