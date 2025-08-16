#pragma once
#include "object_header.h"

struct Object {
    explicit Object(ReaderWriter& io, offset_t pos_)
        : io_(io), file_pos_(pos_)
    {
    }
private:
    ReaderWriter& io_;
    offset_t file_pos_{};
};
