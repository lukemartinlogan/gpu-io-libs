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
    struct Space {
        offset_t offset;
        len_t size;
    };

    [[nodiscard]] std::optional<Space> FindSpace(size_t size, bool must_be_nil) const;

    [[nodiscard]] static std::optional<Space> FindSpaceRecursive(
        Deserializer& de,
        offset_t sb_base_addr,
        uint16_t& messages_read,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint32_t search_size,
        bool must_be_nil
    );

    void JumpToRelativeOffset(offset_t offset) const {
        file_->io.SetPosition(file_pos_ + offset);
    }

private:
    std::shared_ptr<FileLink> file_;
    offset_t file_pos_{};
};
