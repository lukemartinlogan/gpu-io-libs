#pragma once
#include <filesystem>

#include "serialization.h"
#include "../hdf5/types.h"

class StdioWriter : public Serializer {
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

    bool WriteBuffer(std::span<const byte_t> data) final {
        size_t bytes_written = std::fwrite(data.data(), 1, data.size(), file_.get());
        return bytes_written == data.size();
    }

private:
    std::filesystem::path path_;
    std::unique_ptr<FILE, decltype(&std::fclose)> file_;
};

class StdioReader : public Deserializer {
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

    bool ReadBuffer(std::span<byte_t> out) final {
        size_t bytes_read = std::fread(out.data(), 1, out.size(), file_.get());
        return bytes_read == out.size();
    }

    [[nodiscard]] offset_t GetPosition() final {
        const long pos = std::ftell(file_.get());

        if (pos < 0) {
            throw std::runtime_error("failed to get position");
        }

        return static_cast<offset_t>(pos);
    }

    void SetPosition(offset_t offset) final {
        if (std::fseek(file_.get(), static_cast<long>(offset), SEEK_SET) != 0) {
            throw std::runtime_error("Seek failed");
        }
    }

private:
    std::filesystem::path path_;
    std::unique_ptr<FILE, decltype(&std::fclose)> file_;
};

// TODO: is there a way to do this without code duplication
class StdioReaderWriter : public ReaderWriter {
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

    bool WriteBuffer(std::span<const byte_t> data) final {
        size_t bytes_written = std::fwrite(data.data(), 1, data.size(), file_.get());
        return bytes_written == data.size();
    }

    bool ReadBuffer(std::span<byte_t> out) final {
        size_t bytes_read = std::fread(out.data(), 1, out.size(), file_.get());
        return bytes_read == out.size();
    }

    [[nodiscard]] offset_t GetPosition() final {
        const long pos = std::ftell(file_.get());

        if (pos < 0) {
            throw std::runtime_error("failed to get position");
        }

        return static_cast<offset_t>(pos);
    }

    void SetPosition(offset_t offset) final {
        if (std::fseek(file_.get(), static_cast<long>(offset), SEEK_SET) != 0) {
            throw std::runtime_error("Seek failed");
        }
    }

private:
    std::filesystem::path path_;
    std::unique_ptr<FILE, decltype(&std::fclose)> file_;
};