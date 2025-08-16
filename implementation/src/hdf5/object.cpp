#include "object.h"

constexpr uint32_t kPrefixSize = 8;

std::optional<FreeSpace> FindFreeSpaceOfSizeRecursive(Deserializer& de, uint16_t& messages_read, uint16_t total_message_ct, uint32_t size_limit, uint32_t search_size) { // NOLINT(*-no-recursion
    uint32_t bytes_read = 0;

    std::optional<FreeSpace> smallest_found{};

    while (bytes_read < size_limit && messages_read < total_message_ct) {
        auto type = de.Read<uint16_t>();
        auto size_bytes = de.Read<uint16_t>();

        bytes_read += size_bytes + kPrefixSize;
        ++messages_read;

        // flags + reserved
        de.Skip<4>();

        if (type == ObjectHeaderContinuationMessage::kType) {
            auto cont = de.Read<ObjectHeaderContinuationMessage>();

            offset_t return_pos = de.GetPosition();
            de.SetPosition(/* TODO: sb.base_addr + */ cont.offset);

            std::optional<FreeSpace> res = FindFreeSpaceOfSizeRecursive(de, messages_read, total_message_ct, cont.length, search_size);

            if (
                res.has_value() && res->size >= search_size // FIXME: technically the second check is redundant
                && ( !smallest_found.has_value() || res->size < smallest_found->size )
            ) {
                smallest_found = res;
            }

            de.SetPosition(return_pos);
        } else {
            if (type == NilMessage::kType) {
                if (
                    // FIXME
                    size_bytes + (kPrefixSize /* this nil prefix */) - (kPrefixSize /* for new nil prefix */) >= search_size
                    && ( !smallest_found.has_value() || size_bytes < smallest_found->size )
                ) {
                    smallest_found = {
                        .offset = de.GetPosition() - kPrefixSize,
                        .size = size_bytes,
                        .from_nil = true,
                    };
                }
            }

            for (size_t b = 0; b < size_bytes; ++b) {
                de.Skip<byte_t>();
            }
        }
    }

    if (bytes_read < size_limit) {
        if (messages_read < total_message_ct) {
            throw std::runtime_error("redundant check");
        }

        uint32_t remaining_bytes = size_limit - bytes_read;

        if (
            remaining_bytes >= search_size
            && ( !smallest_found.has_value() || remaining_bytes < smallest_found->size )
        ) {
            smallest_found = {
                .offset = de.GetPosition(),
                .size = remaining_bytes,
                .from_nil = false,
            };
        }
    }

    return smallest_found;
}

std::optional<FreeSpace> Object::FindFreeSpaceOfSize(size_t size) const {
    JumpToRelativeOffset(0);

    io_.Skip<2>();

    auto total_message_ct = io_.Read<uint16_t>();

    io_.Skip<4>();

    auto header_size = io_.Read<uint32_t>();

    // reserved
    io_.Skip<4>();

    uint16_t messages_read = 0;

    return FindFreeSpaceOfSizeRecursive(io_, messages_read, total_message_ct, header_size, size);
}
