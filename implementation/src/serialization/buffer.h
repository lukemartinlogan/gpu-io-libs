#pragma once
#include <algorithm>

#include "serialization.h"

class BufferDeserializer {
public:
    explicit BufferDeserializer(cstd::span<const byte_t> buf) // NOLINT
        : buf(buf), cursor(0) {}

    void ReadBuffer(cstd::span<byte_t> out) {
        ASSERT(
            out.size() <= buf.size() - cursor,
            "BufferDeserializer: not enough data in buffer"
        );

        std::copy_n(buf.begin() + static_cast<std::ptrdiff_t>(cursor), out.size(), out.begin());

        cursor += out.size();
    }

    [[nodiscard]] offset_t GetPosition() const {
        return cursor;
    };

    void SetPosition(offset_t offset) {
        cursor = offset;
    }

    [[nodiscard]] bool IsExhausted() const {
        return cursor == buf.size();
    }

    cstd::span<const byte_t> buf;
    size_t cursor;
};

static_assert(serde::Deserializer<BufferDeserializer>);

class BufferReaderWriter {
public:
    explicit BufferReaderWriter(cstd::span<byte_t> buf) // NOLINT
        : buf(buf), cursor(0) {}

    void WriteBuffer(cstd::span<const byte_t> data) {
        ASSERT(
            data.size() <= buf.size() - cursor,
            "BufferReaderWriter: not enough space in buffer for write"
        );

        std::ranges::copy(data, buf.data() + cursor);

        cursor += data.size();
    }

    void ReadBuffer(cstd::span<byte_t> out) {
        ASSERT(
            out.size() <= buf.size() - cursor,
            "BufferReaderWriter: not enough data in buffer for read"
        );

        std::copy_n(buf.begin() + static_cast<std::ptrdiff_t>(cursor), out.size(), out.begin());

        cursor += out.size();
    }

    [[nodiscard]] offset_t GetPosition() const {
        return cursor;
    }

    void SetPosition(offset_t offset) {
        ASSERT(offset <= buf.size(), "BufferReaderWriter: SetPosition out of bounds");
        cursor = offset;
    }

    cstd::span<byte_t> GetWritten() const {
        return buf.subspan(0, cursor);
    }

    [[nodiscard]] bool IsExhausted() const {
        return cursor == buf.size();
    }

    [[nodiscard]] size_t remaining() const {
        return buf.size() - cursor;
    }

    cstd::span<byte_t> buf;
    size_t cursor;
};

static_assert(serde::Serializer<BufferReaderWriter> && serde::Deserializer<BufferReaderWriter>);