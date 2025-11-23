#include <vector>
#include <stdexcept>

#include "local_heap.h"

#include "file_link.h"
#include "../util/string.h"

hdf5::expected<offset_t> LocalHeap::WriteString(hdf5::string_view string, FileLink& file) {
    hdf5::string null_terminated_str(string);

    return WriteBytes(
        cstd::span(
            reinterpret_cast<const byte_t*>(null_terminated_str.c_str()),
            null_terminated_str.size() + 1
        ),
        file
    );
}

cstd::tuple<LocalHeap, offset_t> LocalHeap::AllocateNew(FileLink& file, len_t min_size) {
    len_t aligned_size = std::max(EightBytesAlignedSize(min_size), sizeof(FreeListBlock));

    offset_t heap_offset = file.AllocateAtEOF(kHeaderSize + aligned_size);

    LocalHeap heap;
    heap.this_offset = heap_offset;
    heap.data_segment_address = heap_offset + kHeaderSize;
    heap.data_segment_size = aligned_size;
    heap.free_list_head_offset = 0;

    FreeListBlock new_fl {
        .next_free_list_offset = kLastFreeBlock,
        .size = aligned_size,
    };

    file.io.SetPosition(heap.data_segment_address);
    serde::Write(file.io, new_fl);

    heap.RewriteToFile(file.io);

    return { heap, heap_offset };
}

// note: this method does not rewrite to file
hdf5::expected<void> LocalHeap::ReserveAdditional(FileLink& file, size_t additional_bytes) {
    // 1. determine new size + alloc
    size_t new_size = std::max(
        data_segment_size * 2,
        data_segment_size + additional_bytes
    );

    new_size = EightBytesAlignedSize(new_size);

    offset_t alloc = file.AllocateAtEOF(new_size);

    // 2. move data

    if (data_segment_size > kMaxBufferSizeBytes) {
        return hdf5::error(
            hdf5::HDF5ErrorCode::CapacityExceeded,
            "Local heap data segment too large for bounded buffer"
        );
    }

    cstd::array<byte_t, kMaxBufferSizeBytes> buffer{};
    cstd::span buffer_span(buffer.data(), data_segment_size);

    file.io.SetPosition(data_segment_address);
    file.io.ReadBuffer(buffer_span);

    file.io.SetPosition(alloc);
    file.io.WriteBuffer(buffer_span);

    // TODO: additional bytes are already zeroed since writing to EOF?
    for (len_t i = 0; i < new_size - data_segment_size; ++i) {
        serde::Write(file.io, byte_t{});
    }

    // 4. update free list
    FreeListBlock block{};
    block.size = new_size - data_segment_size;

    if (free_list_head_offset == kUndefinedOffset) {
        block.next_free_list_offset = kLastFreeBlock;
    } else {
        block.next_free_list_offset = free_list_head_offset;
    }

    file.io.SetPosition(alloc + data_segment_size);
    serde::Write(file.io, block);

    free_list_head_offset = data_segment_size;

    // 3. update struct
    data_segment_address = alloc;
    data_segment_size = new_size;

    return {};
}