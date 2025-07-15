#pragma once
#include <filesystem>
#include <fstream>

#include "serialization.h"

class FStreamWriter: Serializer {
public:
    explicit FStreamWriter(const std::filesystem::path& path)
        : path_(path)
    {
        stream_ = std::ofstream(path, std::ofstream::out | std::ofstream::binary);
    }

    ~FStreamWriter() override {
        stream_.close();
    }

    bool WriteBuffer(std::span<const byte_t> data) final {
        stream_.write(reinterpret_cast<const char*>(data.data()), data.size());

        return stream_.good();
    }
private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

class FStreamReader: Deserializer {
public:
    explicit FStreamReader(const std::filesystem::path& path)
        : path_(path)
    {
        stream_ = std::ifstream(path, std::ofstream::in | std::ofstream::binary);
    }

    ~FStreamReader() override {
        stream_.close();
    }

    bool ReadBuffer(std::span<byte_t> out) final {
        stream_.read(reinterpret_cast<char*>(out.data()), out.size());

        return stream_.gcount() == out.size() && stream_.good();
    }
private:
    std::filesystem::path path_;
    std::ifstream stream_;
};