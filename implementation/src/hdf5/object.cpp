#include "object.h"

constexpr uint32_t kPrefixSize = 8;

std::optional<Object::FreeSpace> Object::FindFreeSpaceRecursive(Deserializer& de, offset_t sb_base_addr, uint16_t& messages_read, uint16_t total_message_ct, uint32_t size_limit, uint32_t search_size) { // NOLINT(*-no-recursion
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
            de.SetPosition(sb_base_addr + cont.offset);

            std::optional<FreeSpace> res = FindFreeSpaceRecursive(de, sb_base_addr, messages_read, total_message_ct, cont.length, search_size);

            if (
                res.has_value() && res->size >= search_size // FIXME: technically the second check is redundant
                && ( !smallest_found.has_value() || res->size < smallest_found->size )
            ) {
                smallest_found = res;
            }

            de.SetPosition(return_pos);
        } else {
            if (type == NilMessage::kType) {
                uint16_t total_size = size_bytes + kPrefixSize;

                if (
                    // no new nil header needed || nil header needed
                    total_size == search_size || total_size >= search_size + kPrefixSize
                    && ( !smallest_found.has_value() || total_size < smallest_found->size )
                ) {
                    smallest_found = {
                        .offset = de.GetPosition() - kPrefixSize,
                        .size = total_size,
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

std::optional<Object::FreeSpace> Object::FindFreeSpace(size_t size) const {
    JumpToRelativeOffset(0);

    file_->io.Skip<2>();

    auto total_message_ct = file_->io.Read<uint16_t>();

    file_->io.Skip<4>();

    auto header_size = file_->io.Read<uint32_t>();

    // reserved
    file_->io.Skip<4>();

    uint16_t messages_read = 0;

    return FindFreeSpaceRecursive(file_->io, file_->superblock.base_addr, messages_read, total_message_ct, header_size, size);
}

void WriteHeader(Serializer& s, uint16_t type, uint16_t size, uint8_t flags) {
    s.Write(type);
    s.Write(size);

    s.Write(flags);
    s.Write<std::array<byte_t, 3>>({});
}

std::vector<byte_t> WriteMessageToBuffer(const HeaderMessageVariant& msg) {
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
    return msg_data.buf;
}

void Object::WriteMessage(const HeaderMessageVariant& msg) const {
    std::vector<byte_t> msg_bytes = WriteMessageToBuffer(msg);
    size_t msg_size = msg_bytes.size();

    std::optional<FreeSpace> space = FindFreeSpace(msg_size);

    if (space.has_value()) {
        file_->io.SetPosition(space->offset);
        file_->io.WriteBuffer(msg_bytes);

        if (space->size > msg_size) {
            uint16_t total_nil_size = space->size - msg_size;

            if (total_nil_size < kPrefixSize) {
                throw std::runtime_error("FindFreeSpace didn't return enough size for a nil message header");
            }

            uint16_t nil_size = total_nil_size - kPrefixSize;

            WriteHeader(file_->io, NilMessage::kType, nil_size, 0);
            file_->io.Write(NilMessage { .size = nil_size, });
        }

        JumpToRelativeOffset(2);
        auto ct = file_->io.Read<uint16_t>();
        JumpToRelativeOffset(2);
        file_->io.Write(ct + 1);
    } else {
        size_t cont_size = sizeof(ObjectHeaderContinuationMessage);
        std::optional<FreeSpace> space_cont;

        if (cont_size < msg_size) {
            space_cont = FindFreeSpace(cont_size);
        }

        if (space_cont.has_value()) {
            // put continuation here, allocate new block
        } else {
            // find message to move, allocate new block and move message
        }
    }
}
