#include "object.h"

constexpr uint32_t kPrefixSize = 8;

std::optional<Object::Space> Object::FindSpaceRecursive(  // NOLINT(*-no-recursion
    Deserializer& de,
    offset_t sb_base_addr,
    uint16_t& messages_read,
    uint16_t total_message_ct,
    uint32_t size_limit,
    uint32_t search_size,
    bool must_be_nil
) {
    uint32_t bytes_read = 0;

    std::optional<Space> smallest_found{};

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

            std::optional<Space> res = FindSpaceRecursive(de, sb_base_addr, messages_read, total_message_ct, cont.length, search_size, must_be_nil);

            if (
                res.has_value() && res->size >= search_size // FIXME: technically the second check is redundant
                && ( !smallest_found.has_value() || res->size < smallest_found->size )
            ) {
                smallest_found = res;
            }

            de.SetPosition(return_pos);
        } else {
            if (!must_be_nil || type == NilMessage::kType) {
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

std::optional<Object::Space> Object::FindSpace(size_t size, bool must_be_nil) const {
    JumpToRelativeOffset(0);

    file->io.Skip<2>();

    auto total_message_ct = file->io.Read<uint16_t>();

    file->io.Skip<4>();

    auto header_size = file->io.Read<uint32_t>();

    // reserved
    file->io.Skip<4>();

    uint16_t messages_read = 0;

    return FindSpaceRecursive(file->io, file->superblock.base_addr, messages_read, total_message_ct, header_size, size, must_be_nil);
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

    BufferSerializer prefix_s(msg_data.buf);

    uint16_t kType = std::visit([](const auto& m) { return m.kType; }, msg);

    WriteHeader(prefix_s, kType, msg_size - kPrefixSize, /* FIXME: support flags */ 0);

    if (prefix_s.cursor != kPrefixSize) { // NOLINT: isn't actually always true
        throw std::runtime_error("prefix size was not eight bytes");
    }

    // ReSharper disable once CppDFAUnreachableCode : for some reason, it thinks the cursor prefix size check always throws
    return msg_data.buf;
}

void Object::WriteMessage(const HeaderMessageVariant& msg) const {
    // TODO: chunk message writes
    std::vector<byte_t> msg_bytes = WriteMessageToBuffer(msg);

    std::optional<Space> nil_space = FindSpace(msg_bytes.size(), true);

    JumpToRelativeOffset(2);
    auto written_ct = file->io.Read<uint16_t>();

    if (nil_space.has_value()) {
        // overwriting existing nil message
        written_ct -= 1;

        file->io.SetPosition(nil_space->offset);
        file->io.WriteBuffer(msg_bytes);

        // writing this message
        written_ct += 1;

        if (nil_space->size > msg_bytes.size()) {
            uint16_t total_nil_size = nil_space->size - msg_bytes.size();

            if (total_nil_size < kPrefixSize) {
                throw std::runtime_error("FindFreeSpace didn't return enough size for a nil message header");
            }

            uint16_t nil_size = total_nil_size - kPrefixSize;

            WriteHeader(file->io, NilMessage::kType, nil_size, 0);
            file->io.Write(NilMessage { .size = nil_size, });

            // wrote a nil header
            written_ct += 1;
        }
    } else {
        size_t cont_size = sizeof(ObjectHeaderContinuationMessage) + kPrefixSize;
        std::optional<Space> space_cont;

        if (cont_size < msg_bytes.size()) {
            space_cont = FindSpace(cont_size, true);
        }

        if (space_cont.has_value()) {
            // overwriting this message
            written_ct -= 1;

            // put continuation here, allocate new block
            size_t alloc_space = msg_bytes.size() + cont_size;
            offset_t offset = file->AllocateAtEOF(alloc_space);

            ObjectHeaderContinuationMessage cont {
                .offset = offset,
                .length = alloc_space
            };

            // write object header
            file->io.SetPosition(space_cont->offset);
            WriteHeader(file->io, ObjectHeaderContinuationMessage::kType, sizeof(ObjectHeaderContinuationMessage), /* FIXME(flags) */0);
            file->io.WriteComplex(cont);

            // write cont msg
            written_ct += 1;

            if (space_cont->size > cont_size) {
                uint16_t total_nil_size = space_cont->size - cont_size;

                if (total_nil_size < kPrefixSize) {
                    throw std::runtime_error("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                file->io.Write(NilMessage { .size = nil_size, });

                // writing nil message
                written_ct += 1;
            }

            // write actual data
            file->io.SetPosition(cont.offset);
            file->io.WriteBuffer(msg_bytes);

            // writing data message
            written_ct += 1;

            if (cont.length > msg_bytes.size()) {
                uint16_t total_nil_size = cont.length - msg_bytes.size();

                if (total_nil_size < kPrefixSize) {
                    throw std::runtime_error("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                file->io.WriteComplex(NilMessage { .size = nil_size, });

                // writing nil message
                written_ct += 1;
            }
        } else {
            // overwriting this message
            written_ct -= 1;

            std::optional<Space> space = FindSpace(cont_size, false);

            if (!space.has_value()) {
                throw std::logic_error("there should always be an object header message that can be moved");
            }

            // move the bytes into write buffer
            std::vector<byte_t> moving(space->size);
            file->io.SetPosition(space->offset);
            file->io.ReadBuffer(moving);

            msg_bytes.insert(msg_bytes.end(), moving.begin(), moving.end());

            // new allocation
            size_t alloc_space = msg_bytes.size() + cont_size;
            offset_t offset = file->AllocateAtEOF(alloc_space);

            ObjectHeaderContinuationMessage cont {
                .offset = offset,
                .length = alloc_space
            };

            // write object header
            file->io.SetPosition(space->offset);
            WriteHeader(file->io, ObjectHeaderContinuationMessage::kType, sizeof(ObjectHeaderContinuationMessage), /* FIXME(flags) */0);
            file->io.WriteComplex(cont);

            // writing cont
            written_ct += 1;

            if (space->size > cont_size) {
                uint16_t total_nil_size = space->size - cont_size;

                if (total_nil_size < kPrefixSize) {
                    throw std::runtime_error("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                file->io.WriteComplex(NilMessage { .size = nil_size, });

                // writing nil
                written_ct += 1;
            }

            // write actual data
            file->io.SetPosition(cont.offset);
            file->io.WriteBuffer(msg_bytes);

            // writing data + moved message
            written_ct += 2;

            if (cont.length > msg_bytes.size()) {
                uint16_t total_nil_size = cont.length - msg_bytes.size();

                if (total_nil_size < kPrefixSize) {
                    throw std::runtime_error("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                file->io.WriteComplex(NilMessage { .size = nil_size, });

                // writing nil
                written_ct += 1;
            }
        }
    }

    JumpToRelativeOffset(2);
    file->io.Write(written_ct);
}
