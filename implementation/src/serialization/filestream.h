#pragma once
#include <filesystem>
#include <fstream>

#include "serialization.h"
#include "../hdf5/types.h"

class FStreamWriter: public Serializer {
public:
    explicit FStreamWriter(const std::filesystem::path& path)
        : path_(path), stream_(path, std::ofstream::out | std::ofstream::binary) {}

    bool WriteBuffer(cstd::span<const byte_t> data) final {
        stream_.write(reinterpret_cast<const char*>(data.data()), data.size());

        return stream_.good();
    }
private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

class FStreamReader: public Deserializer {
public:
    explicit FStreamReader(const std::filesystem::path& path)
        : path_(path), stream_(path, std::ofstream::in | std::ofstream::binary) {}

    bool ReadBuffer(cstd::span<byte_t> out) final {
        stream_.read(reinterpret_cast<char*>(out.data()), out.size());

        return stream_.gcount() == out.size() && stream_.good();
    }

    offset_t GetPosition() final {
        const std::streampos pos = stream_.tellg();

        if (pos < 0) {
            throw std::runtime_error("failed to get position");
        }

        return pos;
    }

    void SetPosition(offset_t offset) final {
        stream_.seekg(static_cast<std::streamoff>(offset));

        if (stream_.fail()) {
            throw std::runtime_error("Seek failed");
        }
    }

private:
    std::filesystem::path path_;
    std::ifstream stream_;
};