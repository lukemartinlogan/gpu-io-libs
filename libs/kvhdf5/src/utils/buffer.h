#pragma once

#include "../serde.h"
#include "algorithms.h"

namespace serde {

/**
 * BufferDeserializer - Read-only deserializer from a byte buffer
 * Reads sequentially from a span of const bytes
 */
class BufferDeserializer {
public:
    CROSS_FUN explicit BufferDeserializer(cstd::span<const kvhdf5::byte_t> buf)
        : buf_(buf), cursor_(0) {}

    CROSS_FUN void ReadBuffer(cstd::span<kvhdf5::byte_t> out) {
        KVHDF5_ASSERT(out.size() <= buf_.size() - cursor_,
               "BufferDeserializer: not enough data in buffer");

        algorithms::copy(buf_.subspan(cursor_, out.size()), out);
        cursor_ += out.size();
    }

    CROSS_FUN size_t GetPosition() const {
        return cursor_;
    }

    CROSS_FUN void SetPosition(size_t offset) {
        KVHDF5_ASSERT(offset <= buf_.size(), "BufferDeserializer: SetPosition out of bounds");
        cursor_ = offset;
    }

    CROSS_FUN bool IsExhausted() const {
        return cursor_ == buf_.size();
    }

    CROSS_FUN size_t Remaining() const {
        return buf_.size() - cursor_;
    }

private:
    cstd::span<const kvhdf5::byte_t> buf_;
    size_t cursor_;
};

static_assert(Deserializer<BufferDeserializer>);

/**
 * BufferReaderWriter - Read/write serializer/deserializer from a mutable byte buffer
 * Can both read and write to a span of mutable bytes
 */
class BufferReaderWriter {
public:
    CROSS_FUN explicit BufferReaderWriter(cstd::span<kvhdf5::byte_t> buf)
        : buf_(buf), cursor_(0) {}

    CROSS_FUN void WriteBuffer(cstd::span<const kvhdf5::byte_t> data) {
        KVHDF5_ASSERT(data.size() <= buf_.size() - cursor_,
               "BufferReaderWriter: not enough space in buffer for write");

        algorithms::copy(data, buf_.subspan(cursor_, data.size()));
        cursor_ += data.size();
    }

    CROSS_FUN void ReadBuffer(cstd::span<kvhdf5::byte_t> out) {
        KVHDF5_ASSERT(out.size() <= buf_.size() - cursor_,
               "BufferReaderWriter: not enough data in buffer for read");

        algorithms::copy(buf_.subspan(cursor_, out.size()), out);
        cursor_ += out.size();
    }

    CROSS_FUN size_t GetPosition() const {
        return cursor_;
    }

    CROSS_FUN void SetPosition(size_t offset) {
        KVHDF5_ASSERT(offset <= buf_.size(), "BufferReaderWriter: SetPosition out of bounds");
        cursor_ = offset;
    }

    CROSS_FUN cstd::span<kvhdf5::byte_t> GetWritten() const {
        return buf_.subspan(0, cursor_);
    }

    CROSS_FUN bool IsExhausted() const {
        return cursor_ == buf_.size();
    }

    CROSS_FUN size_t Remaining() const {
        return buf_.size() - cursor_;
    }

private:
    cstd::span<kvhdf5::byte_t> buf_;
    size_t cursor_;
};

static_assert(Serializer<BufferReaderWriter>);
static_assert(Deserializer<BufferReaderWriter>);

} // namespace serde
