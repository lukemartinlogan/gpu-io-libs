#include <vector>
#include <stdexcept>

#include "local_heap.h"

#include "../util/align.h"
#include "file_link.h"


std::string ReadNullTerminatedString(Deserializer& de, cstd::optional<size_t> max_size) {
    std::vector<byte_t> buf;

    while (!max_size || buf.size() < *max_size) {
        auto c = de.Read<byte_t>();

        if (c == static_cast<byte_t>('\0')) {
            break;
        }

        buf.push_back(c);
    }

    return { reinterpret_cast<const char*>(buf.data()), buf.size() };
}

std::string LocalHeap::ReadString(offset_t offset, Deserializer& de) const {
    if (offset >= data_segment_size) {
        throw std::runtime_error("LocalHeap: offset out of bounds");
    }

    auto rem_size = data_segment_size - offset;

    de.SetPosition(data_segment_address + offset);

    return ReadNullTerminatedString(de, rem_size);
}

cstd::optional<LocalHeap::SuitableFreeSpace> LocalHeap::FindFreeSpace(len_t required_size, Deserializer& de) const {
    static_assert(sizeof(FreeListBlock) == 2 * sizeof(len_t), "mismatch between spec");

    if (free_list_head_offset == kUndefinedOffset) {
        return cstd::nullopt;
    }

    offset_t current_offset = free_list_head_offset;
    cstd::optional<offset_t> prev_block_offset = cstd::nullopt;

    while (current_offset != kLastFreeBlock) {
        if (current_offset + sizeof(FreeListBlock) > data_segment_size) {
            throw std::runtime_error("LocalHeap: free list offset out of bounds");
        }

        de.SetPosition(data_segment_address + current_offset);
        auto block = de.ReadRaw<FreeListBlock>();

        if (block.size >= required_size) {
            return SuitableFreeSpace {
                .prev_block_offset = prev_block_offset,
                .this_offset = current_offset,
                .block = block,
            };
        }

        prev_block_offset = current_offset;
        current_offset = block.next_free_list_offset;
    }

    return cstd::nullopt;
}

offset_t LocalHeap::WriteBytes(std::span<const byte_t> data, FileLink& file) {
    len_t aligned_size = EightBytesAlignedSize(data.size());

    auto free = FindFreeSpace(aligned_size, file.io);

    if (!free) {
        ReserveAdditional(file, aligned_size);

        free = FindFreeSpace(aligned_size, file.io);

        if (!free) {
            throw std::logic_error("LocalHeap: failed to allocate space after reserving more");
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
        file.io.SetPosition(data_segment_address + new_block_offset);
        file.io.WriteRaw(new_block);

        if (free->prev_block_offset.has_value()) {
            file.io.SetPosition(data_segment_address + *free->prev_block_offset);
            auto block = file.io.ReadRaw<FreeListBlock>();

            block.next_free_list_offset = new_block_offset;
            file.io.SetPosition(data_segment_address + *free->prev_block_offset);
            file.io.WriteRaw(block);
        } else {
            free_list_head_offset = new_block_offset;
        }
    } else {
        if (free->prev_block_offset.has_value()) {
            file.io.SetPosition(data_segment_address + *free->prev_block_offset);
            auto block = file.io.ReadRaw<FreeListBlock>();

            block.next_free_list_offset = free->block.next_free_list_offset;
            file.io.SetPosition(data_segment_address + *free->prev_block_offset);
            file.io.WriteRaw(block);
        } else {
            if (free->block.next_free_list_offset == kLastFreeBlock) {
                free_list_head_offset = kUndefinedOffset;
            } else {
                free_list_head_offset = free->block.next_free_list_offset;
            }
        }
    }

    file.io.SetPosition(data_segment_address + free->this_offset);
    file.io.WriteBuffer(data);

    for (len_t i = 0; i < aligned_size - data.size(); ++i) {
        file.io.WriteRaw<byte_t>({});
    }

    RewriteToFile(file.io);

    return free->this_offset;
}

offset_t LocalHeap::WriteString(std::string_view string, FileLink& file) {
    std::string null_terminated(string);

    return WriteBytes(
        std::span(
            reinterpret_cast<const byte_t*>(null_terminated.c_str()),
            null_terminated.size() + 1
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
    file.io.WriteRaw(new_fl);

    heap.RewriteToFile(file.io);

    return { heap, heap_offset };
}

// note: this method does not rewrite to file
void LocalHeap::ReserveAdditional(FileLink& file, size_t additional_bytes) {
    // 1. determine new size + alloc
    size_t new_size = std::max(
        data_segment_size * 2,
        data_segment_size + additional_bytes
    );

    new_size = EightBytesAlignedSize(new_size);

    offset_t alloc = file.AllocateAtEOF(new_size);

    // 2. move data
    std::vector<byte_t> buffer(data_segment_size);

    file.io.SetPosition(data_segment_address);
    file.io.ReadBuffer(buffer);

    file.io.SetPosition(alloc);
    file.io.WriteBuffer(buffer);

    // additional bytes are already zeroed since writing to EOF?
    for (len_t i = 0; i < new_size - data_segment_size; ++i) {
        file.io.WriteRaw<byte_t>({});
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
    file.io.WriteRaw(block);

    free_list_head_offset = data_segment_size;

    // 3. update struct
    data_segment_address = alloc;
    data_segment_size = new_size;
}

void LocalHeap::RewriteToFile(ReaderWriter& rw) const {
    rw.SetPosition(this_offset);
    rw.WriteComplex<LocalHeap>(*this);
}

void LocalHeap::Serialize(Serializer& s) const {
    s.Write(kSignature);
    s.Write(kVersionNumber);

    // reserved (zero)
    s.Write<cstd::array<byte_t, 3>>({});

    s.Write<len_t>(data_segment_size);
    s.Write(free_list_head_offset);
    s.Write(data_segment_address);
}

LocalHeap LocalHeap::Deserialize(Deserializer& de) {
    offset_t this_offset = de.GetPosition();

    if (de.Read<cstd::array<uint8_t, 4>>() != kSignature) {
        throw std::runtime_error("Superblock signature was invalid");
    }

    if (de.Read<uint8_t>() != kVersionNumber) {
        throw std::runtime_error("Superblock version number was invalid");
    }
    // reserved (zero)
    de.Skip<uint8_t>();
    de.Skip<uint8_t>();
    de.Skip<uint8_t>();


    LocalHeap heap{};
    heap.data_segment_size = de.Read<len_t>();
    heap.free_list_head_offset = de.Read<len_t>();
    heap.data_segment_address = de.Read<offset_t>();

    heap.this_offset = this_offset;

    return heap;
}