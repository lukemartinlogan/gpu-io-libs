#pragma once
#include <optional>

#include "object_header.h"

struct Object {
    explicit Object(ReaderWriter& io, offset_t pos_)
        : io_(io), file_pos_(pos_)
    {
        JumpToRelativeOffset(0);

        // FIXME: hardcoded constant
        if (io_.Read<uint8_t>() != 0x01) {
            throw std::runtime_error("Version number was invalid");
        }
    }

    void WriteMessage(HeaderMessageVariant msg) const;

private:
    struct FreeSpace {
        offset_t offset;
        len_t size;
        // if this is true, the four byte header of the nil is not counted in size, since it needs to be forwarded
        bool from_nil;
    };

    [[nodiscard]] std::optional<FreeSpace> FindFreeSpaceOfSize(size_t size) const;

    [[nodiscard]] static std::optional<FreeSpace> FindFreeSpaceOfSizeRecursive(
        Deserializer& de,
        uint16_t& messages_read,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint32_t search_size
    );

    void JumpToRelativeOffset(offset_t offset) const {
        io_.SetPosition(file_pos_ + offset);
    }

private:
    ReaderWriter& io_;
    offset_t file_pos_{};
};
