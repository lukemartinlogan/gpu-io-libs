#pragma once
#include "object_header.h"

struct Object {
    explicit Object(Deserializer& de, offset_t pos_)
        : de_(de), file_pos_(pos_) {}
private:
    Deserializer& de_;
    offset_t file_pos_{};
};
