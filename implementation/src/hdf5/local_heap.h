#pragma once

#include <string>
#include <cstdint>

#include "types.h"
#include "../serialization/serialization.h"
#include "gpu_string.h"

struct FileLink;

struct LocalHeap {
    len_t free_list_head_offset{};

    [[nodiscard]] hdf5::expected<hdf5::string> ReadString(offset_t offset, VirtualDeserializer& de) const;

    hdf5::expected<offset_t> WriteString(hdf5::string_view string, FileLink& file);

    static cstd::tuple<LocalHeap, offset_t> AllocateNew(FileLink& file, len_t min_size);

    void RewriteToFile(VirtualReaderWriter& rw) const;

    void Serialize(VirtualSerializer& s) const;

    static hdf5::expected<LocalHeap> Deserialize(VirtualDeserializer& de);

private:
    struct FreeListBlock {
        len_t next_free_list_offset;
        len_t size;
    };

    struct SuitableFreeSpace {
        cstd::optional<offset_t> prev_block_offset;
        offset_t this_offset;
        FreeListBlock block;
    };

    hdf5::expected<cstd::optional<SuitableFreeSpace>> FindFreeSpace(len_t required_size, VirtualDeserializer& de) const;

    hdf5::expected<offset_t> WriteBytes(std::span<const byte_t> data, FileLink& file);

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