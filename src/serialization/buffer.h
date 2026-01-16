#pragma once

#include "serialization.h"
#include "../hdf5/gpu_allocator.h"

// TODO: this class shouldn't have an allocator! instead, maybe a wrapper type?
class BufferDeserializer {
public:
    __device__
    explicit BufferDeserializer(cstd::span<const byte_t> buf) // NOLINT
        : buf(buf), cursor(0), allocator_(nullptr) {}

    __device__
    BufferDeserializer(cstd::span<const byte_t> buf, hdf5::HdfAllocator* alloc)
        : buf(buf), cursor(0), allocator_(alloc) {}

    __device__
    void ReadBuffer(cstd::span<byte_t> out) {
        ASSERT(
            out.size() <= buf.size() - cursor,
            "BufferDeserializer: not enough data in buffer"
        );

        cstd::_copy(
            buf.begin() + static_cast<std::ptrdiff_t>(cursor),
            buf.begin() + static_cast<std::ptrdiff_t>(cursor) + static_cast<std::ptrdiff_t>(out.size()),
            out.begin()
        );

        cursor += out.size();
    }

    __device__
    [[nodiscard]] offset_t GetPosition() const {
        return cursor;
    };

    __device__
    void SetPosition(offset_t offset) {
        cursor = offset;
    }

    __device__
    [[nodiscard]] bool IsExhausted() const {
        return cursor == buf.size();
    }

    __device__
    [[nodiscard]] hdf5::HdfAllocator* GetAllocator() const {
        return allocator_;
    }

    cstd::span<const byte_t> buf;
    size_t cursor;
    hdf5::HdfAllocator* allocator_;
};

static_assert(serde::Deserializer<BufferDeserializer>);
static_assert(iowarp::ProvidesAllocator<BufferDeserializer>);

class BufferReaderWriter {
public:
    __device__
    explicit BufferReaderWriter(cstd::span<byte_t> buf) // NOLINT
        : buf(buf), cursor(0), allocator_(nullptr) {}

    __device__
    BufferReaderWriter(cstd::span<byte_t> buf, hdf5::HdfAllocator* alloc)
        : buf(buf), cursor(0), allocator_(alloc) {}

    __device__
    void WriteBuffer(cstd::span<const byte_t> data) {
        ASSERT(
            data.size() <= buf.size() - cursor,
            "BufferReaderWriter: not enough space in buffer for write"
        );

        cstd::_copy(data.begin(), data.end(), buf.data() + cursor);

        cursor += data.size();
    }

    __device__
    void ReadBuffer(cstd::span<byte_t> out) {
        ASSERT(
            out.size() <= buf.size() - cursor,
            "BufferReaderWriter: not enough data in buffer for read"
        );

        cstd::_copy(buf.begin() + cursor, buf.begin() + cursor + out.size(), out.begin());

        cursor += out.size();
    }

    __device__
    [[nodiscard]] offset_t GetPosition() const {
        return cursor;
    }

    __device__
    void SetPosition(offset_t offset) {
        ASSERT(offset <= buf.size(), "BufferReaderWriter: SetPosition out of bounds");
        cursor = offset;
    }

    __device__
    cstd::span<byte_t> GetWritten() const {
        return buf.subspan(0, cursor);
    }

    __device__
    [[nodiscard]] bool IsExhausted() const {
        return cursor == buf.size();
    }

    __device__
    [[nodiscard]] size_t remaining() const {
        return buf.size() - cursor;
    }

    __device__
    [[nodiscard]] hdf5::HdfAllocator* GetAllocator() const {
        return allocator_;
    }

    cstd::span<byte_t> buf;
    size_t cursor;
    hdf5::HdfAllocator* allocator_;
};

static_assert(serde::Serializer<BufferReaderWriter> && serde::Deserializer<BufferReaderWriter>);
static_assert(iowarp::ProvidesAllocator<BufferReaderWriter>);