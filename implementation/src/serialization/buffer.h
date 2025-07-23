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

        std::copy(data.begin(), data.end(), buf.data() + cursor);

        cursor += data.size();

        return true;
    }

    std::span<byte_t> buf;
    size_t cursor;
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

    std::span<byte_t> buf;
    size_t cursor;
};