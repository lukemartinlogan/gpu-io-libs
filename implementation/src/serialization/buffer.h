#pragma once
#include <algorithm>

#include "serialization.h"

class BufferSerializer {
public:
    explicit BufferSerializer(cstd::span<byte_t> buf) // NOLINT
        : buf(buf), cursor(0) {}

    void WriteBuffer(cstd::span<const byte_t> data) {
        ASSERT(
            data.size() <= buf.size() - cursor,
            "BufferSerializer: not enough space in buffer"
        );

        std::ranges::copy(data, buf.data() + cursor);

        cursor += data.size();
    }

    cstd::span<byte_t> buf;
    size_t cursor;
};

static_assert(serde::Serializer<BufferSerializer>);

class DynamicBufferSerializer {
public:
    explicit DynamicBufferSerializer(size_t size = 0) {
        buf.reserve(size);
    }

    void WriteBuffer(cstd::span<const byte_t> data) {
        buf.insert(buf.end(), data.begin(), data.end());
    }

    std::vector<byte_t> buf;
};

static_assert(serde::Serializer<DynamicBufferSerializer>);

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