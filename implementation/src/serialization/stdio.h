#pragma once
#include <filesystem>

#include "serialization.h"

class StdioWriter : public Serializer {
public:
    explicit StdioWriter(const std::filesystem::path& path)
        : path_(path)
    {
        file_ = std::fopen(path.string().c_str(), "wb"); // NOLINT

        if (!file_) {
            throw std::runtime_error("failed to open file");
        }
    }

    ~StdioWriter() override {
        if (file_) {
            std::fclose(file_);
        }
    }

    bool WriteBuffer(std::span<const byte_t> data) final {
        size_t bytes_written = std::fwrite(data.data(), 1, data.size(), file_);
        return bytes_written == data.size();
    }

private:
    std::filesystem::path path_;
    FILE* file_;
};

class StdioReader : public Deserializer {
public:
    explicit StdioReader(const std::filesystem::path& path)
        : path_(path)
    {
        file_ = std::fopen(path.string().c_str(), "rb"); // NOLINT

        if (!file_) {
            throw std::runtime_error("failed to open file");
        }
    }

    ~StdioReader() override {
        if (file_) {
            std::fclose(file_);
        }
    }

    bool ReadBuffer(std::span<byte_t> out) final {
        size_t bytes_read = std::fread(out.data(), 1, out.size(), file_);
        return bytes_read == out.size();
    }

private:
    std::filesystem::path path_;
    FILE* file_;
};