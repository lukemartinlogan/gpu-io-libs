#include "object.h"

#include "../util/align.h"

cstd::optional<Object::Space> Object::FindSpace(size_t size, bool must_be_nil) const {
    JumpToRelativeOffset(0);

    auto& io = file->io;

    serde::Skip(io, 2);

    auto total_message_ct = serde::Read<decltype(io), uint16_t>(io);

    serde::Skip(io, 4);

    auto header_size = serde::Read<decltype(io), uint32_t>(io);

    // reserved
    serde::Skip(io, 4);

    uint16_t messages_read = 0;

    return FindSpaceRecursive(io, file->superblock.base_addr, messages_read, total_message_ct, header_size, size, must_be_nil);
}

std::vector<byte_t> WriteMessageToBuffer(const HeaderMessageVariant& msg) {
    DynamicBufferSerializer msg_data;

    // reserve eight bytes for prefix
    serde::Write(msg_data, cstd::array<byte_t, kPrefixSize>{}); // reserved

    cstd::visit([&msg_data](const auto& m) { serde::Write(msg_data, m); }, msg);

    while (msg_data.buf.size() % 8 != 0) {
        serde::Write(msg_data, byte_t{});
    }

    size_t msg_size = msg_data.buf.size();

    BufferSerializer prefix_s(msg_data.buf);

    uint16_t kType = cstd::visit([](const auto& m) { return m.kType; }, msg);

    WriteHeader(prefix_s, kType, msg_size - kPrefixSize, /* FIXME: support flags */ 0);

    ASSERT(prefix_s.cursor == kPrefixSize, "prefix size was not eight bytes");

    // ReSharper disable once CppDFAUnreachableCode : for some reason, it thinks the cursor prefix size check always throws
    return msg_data.buf;
}

void Object::WriteMessage(const HeaderMessageVariant& msg) const {
    // TODO: chunk message writes
    std::vector<byte_t> msg_bytes = WriteMessageToBuffer(msg);

    cstd::optional<Space> nil_space = FindSpace(msg_bytes.size(), true);

    JumpToRelativeOffset(2);
    auto written_ct = serde::Read<decltype(file->io), uint16_t>(file->io);

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
                UNREACHABLE("FindFreeSpace didn't return enough size for a nil message header");
            }

            uint16_t nil_size = total_nil_size - kPrefixSize;

            WriteHeader(file->io, NilMessage::kType, nil_size, 0);
            serde::Write(file->io, NilMessage { .size = nil_size, })

            // wrote a nil header
            written_ct += 1;
        }

        ASSERT(file->io.GetPosition() <= nil_space->offset + nil_space->size, "wrote more bytes than the nil space allowed");
    } else {
        size_t cont_size = sizeof(ObjectHeaderContinuationMessage) + kPrefixSize;
        cstd::optional<Space> space_cont;

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
            serde::Write(file->io, cont);

            // write cont msg
            written_ct += 1;

            if (space_cont->size > cont_size) {
                uint16_t total_nil_size = space_cont->size - cont_size;

                if (total_nil_size < kPrefixSize) {
                    UNREACHABLE("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                serde::Write(file->io, NilMessage { .size = nil_size, });

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
                    UNREACHABLE("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                serde::Write(file->io, NilMessage { .size = nil_size, });

                // writing nil message
                written_ct += 1;
            }
        } else {
            // overwriting this message
            written_ct -= 1;

            cstd::optional<Space> space = FindSpace(cont_size, false);

            ASSERT(space.has_value(), "there should always be an object header message that can be moved");

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
            serde::Write(file->io, cont);

            // writing cont
            written_ct += 1;

            if (space->size > cont_size) {
                uint16_t total_nil_size = space->size - cont_size;

                if (total_nil_size < kPrefixSize) {
                    UNREACHABLE("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                serde::Write(file->io, NilMessage { .size = nil_size, });

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
                    UNREACHABLE("FindFreeSpace didn't return enough size for a nil message header");
                }

                uint16_t nil_size = total_nil_size - kPrefixSize;

                WriteHeader(file->io, NilMessage::kType, nil_size, 0);
                serde::Write(file->io, NilMessage { .size = nil_size, });

                // writing nil
                written_ct += 1;
            }
        }
    }

    JumpToRelativeOffset(2);
    serde::Write(file->io, written_ct);
}

// semantically, this isn't const, so it's not being made const
// ReSharper disable once CppMemberFunctionMayBeConst
cstd::optional<ObjectHeaderMessage> Object::DeleteMessage(uint16_t msg_type) {
    JumpToRelativeOffset(0);

    serde::Skip(file->io, 2);


    auto total_message_ct = serde::Read<decltype(file->io), uint16_t>(file->io);

    serde::Skip(file->io, 4);

    auto header_size = serde::Read<decltype(file->io), uint32_t>(file->io);

    // reserved
    serde::Skip(file->io, 4);

    uint16_t messages_read = 0;

    cstd::optional<Space> found = FindMessageRecursive(file->io, file->superblock.base_addr, messages_read, total_message_ct, header_size, msg_type);

    if (!found.has_value()) {
        return cstd::nullopt;
    }

    file->io.SetPosition(found->offset);
    auto msg_result = serde::Read<decltype(file->io), ObjectHeaderMessage>(file->io);


    // TODO(refactor-exceptions): this method should return an expected
    if (!msg_result) {
        return cstd::nullopt;
    }

    ASSERT(file->io.GetPosition() <= found->offset + found->size, "Read too many bytes for message");

    file->io.SetPosition(found->offset);
    uint16_t nil_size = found->size - kPrefixSize;

    WriteHeader(file->io, NilMessage::kType, nil_size, 0);
    serde::Write(file->io, NilMessage { .size = nil_size, });

    return *msg_result;
}

// TODO: fix code duplication
cstd::optional<ObjectHeaderMessage> Object::GetMessage(uint16_t msg_type) {
    JumpToRelativeOffset(0);

    serde::Skip(file->io, 2);

    auto total_message_ct = serde::Read<decltype(file->io), uint16_t>(file->io);


    serde::Skip(file->io, 4);

    auto header_size = serde::Read<decltype(file->io), uint32_t>(file->io);

    // reserved
    serde::Skip(file->io, 4);

    uint16_t messages_read = 0;

    cstd::optional<Space> found = FindMessageRecursive(file->io, file->superblock.base_addr, messages_read, total_message_ct, header_size, msg_type);

    if (!found.has_value()) {
        return cstd::nullopt;
    }

    file->io.SetPosition(found->offset);
    auto msg_result = serde::Read<decltype(file->io), ObjectHeaderMessage>(file->io);

    if (!msg_result) {
        return cstd::nullopt;
    }

    ASSERT(file->io.GetPosition() <= found->offset + found->size, "Read too many bytes for message");

    return *msg_result;
}

Object Object::AllocateEmptyAtEOF(len_t min_size, const std::shared_ptr<FileLink>& file) {
    len_t alloc_size = EmptyHeaderMessagesSize(min_size) + 16;

    offset_t alloc_start = file->AllocateAtEOF(alloc_size);
    file->io.SetPosition(alloc_start);
    WriteEmpty(min_size, file->io);

    len_t bytes_written = file->io.GetPosition() - alloc_start;

    ASSERT(bytes_written == alloc_size, "AllocateEmptyAtEOF: size mismatch");

    return Object(file, alloc_start);
}
