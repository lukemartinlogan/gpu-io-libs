#pragma once

#include <string>
#include <cstdint>

#include "types.h"
#include "../serialization/serialization.h"
#include "gpu_string.h"
#include "../util/align.h"
#include "../util/string.h"

struct FileLink;

struct LocalHeap {
    len_t free_list_head_offset{};

    // TODO(cuda_vector): many calls of this function don't need ownership; 'LocalHeap::ViewString'?
    template<serde::Deserializer D>
    [[nodiscard]] hdf5::expected<hdf5::string> ReadString(offset_t offset, D& de) const {
        if (offset >= data_segment_size) {
            return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "LocalHeap: offset out of bounds");
        }

        auto rem_size = data_segment_size - offset;

        de.SetPosition(data_segment_address + offset);

        return ReadNullTerminatedString(de, rem_size);
    }

    hdf5::expected<offset_t> WriteString(hdf5::string_view string, FileLink& file);

    static cstd::tuple<LocalHeap, offset_t> AllocateNew(FileLink& file, len_t min_size);

    template<serde::Serializer S> requires serde::Seekable<S>
    void RewriteToFile(S& s) const {
        s.SetPosition(this_offset);
        serde::Write(s, *this);
    }

    template<serde::Serializer S>
    void Serialize(S& s) const {
        serde::Write(s, kSignature);
        serde::Write(s, kVersionNumber);

        // reserved (zero)
        serde::Write(s, cstd::array<byte_t, 3>{});

        serde::Write(s, data_segment_size);
        serde::Write(s, free_list_head_offset);
        serde::Write(s, data_segment_address);
    }

    template<serde::Deserializer D>
    static hdf5::expected<LocalHeap> Deserialize(D& de) {
        offset_t this_offset = de.GetPosition();

        if (serde::Read<D, cstd::array<uint8_t, 4>>(de) != kSignature) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "LocalHeap signature was invalid");
        }

        if (serde::Read<D, uint8_t>(de) != kVersionNumber) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "LocalHeap version number was invalid");
        }
        // reserved (zero)
        serde::Skip<D, uint8_t>(de);
        serde::Skip<D, uint8_t>(de);
        serde::Skip<D, uint8_t>(de);

        LocalHeap heap{};
        heap.data_segment_size = serde::Read<D, len_t>(de);
        heap.free_list_head_offset = serde::Read<D, len_t>(de);
        heap.data_segment_address = serde::Read<D, offset_t>(de);

        heap.this_offset = this_offset;

        return heap;
    }

private:
    struct FreeListBlock {
        len_t next_free_list_offset;
        len_t size;
    };

    static_assert(serde::TriviallySerializable<FreeListBlock>);

    struct SuitableFreeSpace {
        cstd::optional<offset_t> prev_block_offset;
        offset_t this_offset;
        FreeListBlock block;
    };

    template<serde::Deserializer D>
    hdf5::expected<cstd::optional<SuitableFreeSpace>> FindFreeSpace(len_t required_size, D& de) const {
        static_assert(sizeof(FreeListBlock) == 2 * sizeof(len_t), "mismatch between spec");

        if (free_list_head_offset == kUndefinedOffset) {
            return cstd::nullopt;
        }

        offset_t current_offset = free_list_head_offset;
        cstd::optional<offset_t> prev_block_offset = cstd::nullopt;

        while (current_offset != kLastFreeBlock) {
            if (current_offset + sizeof(FreeListBlock) > data_segment_size) {
                return hdf5::error(hdf5::HDF5ErrorCode::IndexOutOfBounds, "LocalHeap: free list offset out of bounds");
            }

            de.SetPosition(data_segment_address + current_offset);
            auto block = serde::Read<D, FreeListBlock>(de);

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

    hdf5::expected<offset_t> WriteBytes(cstd::span<const byte_t> data, FileLink& file);

    hdf5::expected<void> ReserveAdditional(FileLink& file, size_t additional_bytes);

private:
    offset_t data_segment_address{};
    len_t data_segment_size{};

    offset_t this_offset{};

    static constexpr offset_t kLastFreeBlock = 1;
    static constexpr len_t kHeaderSize = 32;
    static constexpr size_t kMaxBufferSizeBytes = 2048;

    static constexpr cstd::array<uint8_t, 4> kSignature = { 'H', 'E', 'A', 'P' };
    static constexpr uint8_t kVersionNumber = 0x00;
};