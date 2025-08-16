#pragma once
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

private:
    void JumpToRelativeOffset(offset_t offset) const {
        io_.SetPosition(file_pos_ + offset);
    }

    ReaderWriter& io_;
    offset_t file_pos_{};
};
