#pragma once
#include <algorithm>

#include "serialization.h"

class BufferSerializer : public Serializer {
public:
    explicit BufferSerializer(std::span<byte_t> buf) // NOLINT
        : buf(buf), cursor(0) {}

    bool WriteBuffer(std::span<const byte_t> data) final {
        if (data.size() > buf.size() - cursor) {
            // not enough remaining space
            return false;
        }

        std::ranges::copy(data, buf.data() + cursor);

        cursor += data.size();

        return true;
    }

    std::span<byte_t> buf;
    size_t cursor;
};

class DynamicBufferSerializer : public Serializer {
public:
    explicit DynamicBufferSerializer(size_t size = 0) {
        buf.reserve(size);
    }

    bool WriteBuffer(std::span<const byte_t> data) final {
        buf.insert(buf.end(), data.begin(), data.end());
        // FIXME: no error reporting
        return true;
    }

    std::vector<byte_t> buf;
};

class BufferDeserializer : public Deserializer {
public:
    BufferDeserializer(std::span<byte_t> buf) // NOLINT
        : buf(buf), cursor(0) {}

    bool ReadBuffer(std::span<byte_t> out) final {
        if (out.size() > buf.size() - cursor) {
            // not enough remaining data
            return false;
        }

        // FIXME: correctly cast size
        std::copy_n(buf.begin() + cursor, out.size(), out.begin());

        cursor += out.size();

        return true;
    }

    [[nodiscard]] offset_t GetPosition() final {
        return cursor;
    };

    void SetPosition(offset_t offset) final {
        cursor = offset;
    }

    [[nodiscard]] bool IsExhausted() const {
        return cursor == buf.size();
    }

    std::span<byte_t> buf;
    size_t cursor;
};