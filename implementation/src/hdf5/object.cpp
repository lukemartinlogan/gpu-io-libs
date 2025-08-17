#include "object.h"

constexpr uint32_t kPrefixSize = 8;

std::optional<Object::FreeSpace> Object::FindFreeSpaceOfSizeRecursive(Deserializer& de, uint16_t& messages_read, uint16_t total_message_ct, uint32_t size_limit, uint32_t search_size) { // NOLINT(*-no-recursion
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

    return smallest_found;
}

std::optional<Object::FreeSpace> Object::FindFreeSpaceOfSize(size_t size) const {
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

void WriteHeader(Serializer& s, uint16_t type, uint16_t size, uint8_t flags) {
    s.Write(type);
    s.Write(size);

    s.Write(flags);
    s.Write<std::array<byte_t, 3>>({});
}

void Object::WriteMessage(HeaderMessageVariant msg) const {
    DynamicBufferSerializer msg_data;

    // reserve eight bytes for prefix
    msg_data.Write<std::array<byte_t, kPrefixSize>>({});

    std::visit([&msg_data](const auto& m) { msg_data.WriteComplex(m); }, msg);

    while (msg_data.buf.size() % 8 != 0) {
        msg_data.Write<uint8_t>(0);
    }

    size_t msg_size = msg_data.buf.size();

    {
        BufferSerializer prefix_s(msg_data.buf);

        uint16_t kType = std::visit([](const auto& m) { return m.kType; }, msg);
        uint16_t size = msg_size - 8;

        WriteHeader(prefix_s, kType, size, /* FIXME: support flags */ 0);

        if (prefix_s.cursor != kPrefixSize) { // NOLINT: isn't actually always true
            throw std::runtime_error("prefix size was not eight bytes");
        }
    }

    // ReSharper disable once CppDFAUnreachableCode : for some reason, it thinks the cursor prefix size check always throws
    std::optional<FreeSpace> space = FindFreeSpaceOfSize(msg_size);

    if (space.has_value()) {
        io_.SetPosition(space->offset);
        io_.WriteBuffer(msg_data.buf);

        // FIXME: make sure to read and write the extra eight bytes before the data
        // FIXME: consume nil header, consecutive nil
        if (space->from_nil) {
            uint16_t nil_size = space->size - msg_size;

            WriteHeader(io_, NilMessage::kType, nil_size, 0);
            msg_data.Write(NilMessage { .size = nil_size, });
        }

        JumpToRelativeOffset(2);
        auto ct = io_.Read<uint16_t>();
        JumpToRelativeOffset(2);
        io_.Write(ct + 1);
    } else {
        throw std::runtime_error("no free space found for message");
    }
}
