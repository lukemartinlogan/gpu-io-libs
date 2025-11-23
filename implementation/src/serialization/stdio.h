#pragma once
#include <filesystem>
#include <functional>

#include "serialization.h"
#include "../hdf5/types.h"

class StdioWriter {
public:
    explicit StdioWriter(const std::filesystem::path& path)
        : path_(path), file_(nullptr, &std::fclose)
    {
        FILE* raw = std::fopen(path.string().c_str(), "wb"); // NOLINT

        if (!raw) {
            throw std::runtime_error("failed to open file");
        }

        file_.reset(raw);
    }

    void WriteBuffer(cstd::span<const byte_t> data) {
        size_t bytes_written = std::fwrite(data.data(), 1, data.size(), file_.get());

        ASSERT(bytes_written == data.size(), "failed to write all bytes to file");
    }

private:
    std::filesystem::path path_;
    std::unique_ptr<FILE, std::function<int(FILE*)>> file_;
};

static_assert(serde::Serializer<StdioWriter>);

class StdioReader {
public:
    explicit StdioReader(const std::filesystem::path& path)
        : path_(path), file_(nullptr, &std::fclose)
    {
        FILE* raw = std::fopen(path.string().c_str(), "rb"); // NOLINT

        if (!raw) {
            throw std::runtime_error("failed to open file");
        }

        file_.reset(raw);
    }

    void ReadBuffer(cstd::span<byte_t> out) {
        size_t bytes_read = std::fread(out.data(), 1, out.size(), file_.get());

        ASSERT(bytes_read == out.size(), "failed to read all bytes from file");
    }

    [[nodiscard]] offset_t GetPosition() const {
        const long pos = std::ftell(file_.get());

        ASSERT(pos >= 0, "failed to get position");

        return static_cast<offset_t>(pos);
    }

    void SetPosition(offset_t offset) {
        ASSERT(std::fseek(file_.get(), static_cast<long>(offset), SEEK_SET) == 0, "seek failed");
    }

private:
    std::filesystem::path path_;
    std::unique_ptr<FILE, std::function<int(FILE*)>> file_;
};

static_assert(serde::Deserializer<StdioReader>);

// TODO: is there a way to do this without code duplication
class StdioReaderWriter {
public:
    explicit StdioReaderWriter(const std::filesystem::path& path)
    : path_(path), file_(nullptr, &std::fclose)
    {
        FILE* raw = std::fopen(path.string().c_str(), "r+b"); // NOLINT

        if (!raw) {
            throw std::runtime_error("failed to open file");
        }

        file_.reset(raw);
    }

    void WriteBuffer(cstd::span<const byte_t> data) const {
        size_t bytes_written = std::fwrite(data.data(), 1, data.size(), file_.get());

        ASSERT(bytes_written == data.size(), "failed to write all bytes to file");
    }

    void ReadBuffer(cstd::span<byte_t> out) const {
        size_t bytes_read = std::fread(out.data(), 1, out.size(), file_.get());

        ASSERT(bytes_read == out.size(), "failed to read all bytes from file");
    }

    [[nodiscard]] offset_t GetPosition() const {
        const long pos = std::ftell(file_.get());

        ASSERT(pos >= 0, "failed to get position");

        return static_cast<offset_t>(pos);
    }

    void SetPosition(offset_t offset) const {
        ASSERT(std::fseek(file_.get(), static_cast<long>(offset), SEEK_SET) == 0, "seek failed");
    }

private:
    std::filesystem::path path_;
    std::unique_ptr<FILE, std::function<int(FILE*)>> file_;
};

static_assert(serde::Serializer<StdioReaderWriter> && serde::Deserializer<StdioReaderWriter>);