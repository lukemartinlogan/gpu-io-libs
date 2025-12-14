#include <vector>
#include <stdexcept>

#include "local_heap.h"

#include "file_link.h"
#include "../util/string.h"

__device__ __host__
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

__device__ __host__
cstd::tuple<LocalHeap, offset_t> LocalHeap::AllocateNew(FileLink& file, len_t min_size) {
    // TODO: windows defines max as a macro :(
    len_t aligned_size = EightBytesAlignedSize(min_size) > sizeof(FreeListBlock) ? EightBytesAlignedSize(min_size) : sizeof(FreeListBlock);

    auto io = file.MakeRW();
    offset_t heap_offset = file.AllocateAtEOF(kHeaderSize + aligned_size, io);

    LocalHeap heap;
    heap.this_offset = heap_offset;
    heap.data_segment_address = heap_offset + kHeaderSize;
    heap.data_segment_size = aligned_size;
    heap.free_list_head_offset = 0;

    FreeListBlock new_fl {
        .next_free_list_offset = kLastFreeBlock,
        .size = aligned_size,
    };

    io.SetPosition(heap.data_segment_address);
    serde::Write(io, new_fl);

    heap.RewriteToFile(io);

    return { heap, heap_offset };
}

__device__ __host__
hdf5::expected<offset_t> LocalHeap::WriteBytes(cstd::span<const byte_t> data, FileLink& file) {
    len_t aligned_size = EightBytesAlignedSize(data.size());

    auto io = file.MakeRW();
    auto free_result = FindFreeSpace(aligned_size, io);
    if (!free_result) {
        return cstd::unexpected(free_result.error());
    }
    auto free = *free_result;

    if (!free) {
        if (auto reserve_result = ReserveAdditional(file, aligned_size); !reserve_result) {
            return cstd::unexpected(reserve_result.error());
        }

        free_result = FindFreeSpace(aligned_size, io);
        if (!free_result) {
            return cstd::unexpected(free_result.error());
        }
        free = *free_result;

        if (!free) {
            return hdf5::error(hdf5::HDF5ErrorCode::AllocationMismatch, "LocalHeap: failed to allocate space after reserving more");
        }
    }

    if (
        free->block.size > aligned_size &&
        free->block.size - aligned_size >= sizeof(FreeListBlock)
    ) {
        len_t remaining_size = free->block.size - aligned_size;

        FreeListBlock new_block {
            .next_free_list_offset = free->block.next_free_list_offset,
            .size = remaining_size,
        };

        offset_t new_block_offset = free->this_offset + aligned_size;
        io.SetPosition(data_segment_address + new_block_offset);
        serde::Write(io, new_block);

        if (free->prev_block_offset.has_value()) {
            io.SetPosition(data_segment_address + *free->prev_block_offset);
            auto block = serde::Read<FreeListBlock>(io);

            block.next_free_list_offset = new_block_offset;
            io.SetPosition(data_segment_address + *free->prev_block_offset);
            serde::Write(io, block);
        } else {
            free_list_head_offset = new_block_offset;
        }
    } else {
        if (free->prev_block_offset.has_value()) {
            io.SetPosition(data_segment_address + *free->prev_block_offset);
            auto block = serde::Read<FreeListBlock>(io);

            block.next_free_list_offset = free->block.next_free_list_offset;
            io.SetPosition(data_segment_address + *free->prev_block_offset);
            serde::Write(io, block);
        } else {
            if (free->block.next_free_list_offset == kLastFreeBlock) {
                free_list_head_offset = kUndefinedOffset;
            } else {
                free_list_head_offset = free->block.next_free_list_offset;
            }
        }
    }

    io.SetPosition(data_segment_address + free->this_offset);
    io.WriteBuffer(data);

    for (len_t i = 0; i < aligned_size - data.size(); ++i) {
        serde::Write(io, byte_t{});
    }

    RewriteToFile(io);

    return free->this_offset;
}

// note: this method does not rewrite to file
__device__ __host__
hdf5::expected<void> LocalHeap::ReserveAdditional(FileLink& file, size_t additional_bytes) {
    // 1. determine new size + alloc
    // TODO: windows defines max as a macro :(
    size_t new_size = data_segment_size * 2 > data_segment_size + additional_bytes ? data_segment_size * 2 : data_segment_size + additional_bytes;

    new_size = EightBytesAlignedSize(new_size);

    auto io = file.MakeRW();
    offset_t alloc = file.AllocateAtEOF(new_size, io);

    // 2. move data

    if (data_segment_size > kMaxBufferSizeBytes) {
        return hdf5::error(
            hdf5::HDF5ErrorCode::CapacityExceeded,
            "Local heap data segment too large for bounded buffer"
        );
    }

    cstd::array<byte_t, kMaxBufferSizeBytes> buffer{};
    cstd::span buffer_span(buffer.data(), data_segment_size);

    io.SetPosition(data_segment_address);
    io.ReadBuffer(buffer_span);

    io.SetPosition(alloc);
    io.WriteBuffer(buffer_span);

    // TODO: additional bytes are already zeroed since writing to EOF?
    for (len_t i = 0; i < new_size - data_segment_size; ++i) {
        serde::Write(io, byte_t{});
    }

    // 4. update free list
    FreeListBlock block{};
    block.size = new_size - data_segment_size;

    if (free_list_head_offset == kUndefinedOffset) {
        block.next_free_list_offset = kLastFreeBlock;
    } else {
        block.next_free_list_offset = free_list_head_offset;
    }

    io.SetPosition(alloc + data_segment_size);
    serde::Write(io, block);

    free_list_head_offset = data_segment_size;

    // 3. update struct
    data_segment_address = alloc;
    data_segment_size = new_size;

    return {};
}