#pragma once
#include <optional>

#include "file_link.h"
#include "object_header.h"

struct Object {
    explicit Object(const std::shared_ptr<FileLink>& file, offset_t pos_)
        : file_(file), file_pos_(pos_)
    {
        JumpToRelativeOffset(0);

        // FIXME: hardcoded constant
        if (file_->io.Read<uint8_t>() != 0x01) {
            throw std::runtime_error("Version number was invalid");
        }
    }

    void WriteMessage(const HeaderMessageVariant& msg) const;

private:
    struct FreeSpace {
        offset_t offset;
        len_t size;
        // if this is true, the four byte header of the nil is not counted in size, since it needs to be forwarded
        bool from_nil;
    };

    [[nodiscard]] std::optional<FreeSpace> FindFreeSpace(size_t size) const;

    [[nodiscard]] static std::optional<FreeSpace> FindFreeSpaceRecursive(
        Deserializer& de,
        offset_t sb_base_addr,
        uint16_t& messages_read,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint32_t search_size
    );

    void JumpToRelativeOffset(offset_t offset) const {
        file_->io.SetPosition(file_pos_ + offset);
    }

private:
    std::shared_ptr<FileLink> file_;
    offset_t file_pos_{};
};
