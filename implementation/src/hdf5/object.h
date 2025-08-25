#pragma once
#include <optional>

#include "file_link.h"
#include "object_header.h"

// TODO: create iterator over messages
struct Object {
    // FIXME: get rid of this ctor
    Object() = default;

    explicit Object(const std::shared_ptr<FileLink>& file, offset_t pos_)
        : file(file), file_pos_(pos_)
    {
        JumpToRelativeOffset(0);

        // FIXME: hardcoded constant
        if (file->io.Read<uint8_t>() != 0x01) {
            throw std::runtime_error("Version number was invalid");
        }
    }

    ObjectHeader GetHeader() const {
        JumpToRelativeOffset(0);

        return file->io.Read<ObjectHeader>();
    }

    [[nodiscard]] offset_t GetAddress() const {
        return file_pos_;
    }

    // TODO: should this mutate an internally held object as well?
    // TODO: add a 'dirty' field to header messages
    void WriteMessage(const HeaderMessageVariant& msg) const;

    static void WriteEmpty(len_t min_size, Serializer& s);

    static Object AllocateEmptyAtEOF(len_t min_size, const std::shared_ptr<FileLink>& file);

public:
    std::shared_ptr<FileLink> file;

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
        file->io.SetPosition(file_pos_ + offset);
    }

private:
    offset_t file_pos_{};
};
