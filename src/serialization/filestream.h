#pragma once
#include <filesystem>
#include <fstream>

#include "serialization.h"
#include "../hdf5/types.h"

class FStreamWriter {
public:
    explicit FStreamWriter(const std::filesystem::path& path)
        : path_(path), stream_(path, std::ofstream::out | std::ofstream::binary) {}

    void WriteBuffer(cstd::span<const byte_t> data) {
        stream_.write(reinterpret_cast<const char*>(data.data()), data.size());

        ASSERT(stream_.good(), "failed to write all bytes to file");
    }
private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

static_assert(serde::Serializer<FStreamWriter>);

class FStreamReader {
public:
    explicit FStreamReader(const std::filesystem::path& path)
        : path_(path), stream_(path, std::ofstream::in | std::ofstream::binary) {}

    void ReadBuffer(cstd::span<byte_t> out) {
        stream_.read(reinterpret_cast<char*>(out.data()), out.size());

        ASSERT(
            stream_.gcount() == out.size() && stream_.good(),
            "failed to read all bytes from file"
        );
    }

    offset_t GetPosition() {
        const std::streampos pos = stream_.tellg();


        ASSERT(pos >= 0, "failed to get position");

        return pos;
    }

    void SetPosition(offset_t offset) {
        stream_.seekg(static_cast<std::streamoff>(offset));

        ASSERT(!stream_.fail(), "seek failed");
    }

private:
    std::filesystem::path path_;
    std::ifstream stream_;
};

static_assert(serde::Deserializer<FStreamReader>);