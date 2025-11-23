#pragma once

#include "file_link.h"
#include "types.h"
#include "object_header.h"

// TODO: create iterator over messages
class Object {
public:
    // FIXME: get rid of this ctor
    Object() = default;

    static hdf5::expected<Object> New(const std::shared_ptr<FileLink>& file, offset_t pos_) {
        file->io.SetPosition(pos_);

        // FIXME: hardcoded constant
        if (serde::Read<decltype(file->io), uint8_t>(file->io) != 0x01) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Object version number was invalid");
        }

        return Object(file, pos_);
    }

    [[nodiscard]] hdf5::expected<ObjectHeader> GetHeader() const {
        JumpToRelativeOffset(0);

        return serde::Read<decltype(file->io), ObjectHeader>(file->io);
    }

    [[nodiscard]] offset_t GetAddress() const {
        return file_pos_;
    }

    // TODO: should this mutate an internally held object as well?
    // TODO: add a 'dirty' field to header messages
    void WriteMessage(const HeaderMessageVariant& msg) const;

    cstd::optional<ObjectHeaderMessage> DeleteMessage(uint16_t msg_type);

    template<typename T>
    cstd::optional<T> DeleteMessage() {
        cstd::optional<ObjectHeaderMessage> msg = DeleteMessage(T::kType);

        if (msg.has_value()) {
            return cstd::get<T>(msg->message);
        } else {
            return cstd::nullopt;
        }
    }

    cstd::optional<ObjectHeaderMessage> GetMessage(uint16_t msg_type);

    template<typename T>
    cstd::optional<T> GetMessage() {
        cstd::optional<ObjectHeaderMessage> msg = GetMessage(T::kType);

        if (msg.has_value()) {
            return cstd::get<T>(msg->message);
        } else {
            return cstd::nullopt;
        }
    }

    template<serde::Serializer S>
    static void WriteEmpty(len_t min_size, S& s) {
        // TODO: this probably shouldn't be unused!
        len_t aligned_size = EmptyHeaderMessagesSize(min_size);

        serde::Write(s, ObjectHeader::kVersionNumber);

        // reserved
        serde::Write<S, uint8_t>(s, 0);
        // total num of messages (one nil message)
        serde::Write<S, uint16_t>(s, 1);

        // object ref count
        serde::Write<S, uint32_t>(s, 0);
        // header size
        serde::Write<S, uint32_t>(s, min_size);

        // reserved
        serde::Write<S, uint32_t>(s, 0);

        // TODO: fix size overflow?
        uint16_t nil_size = min_size - 8;

        WriteHeader(s, NilMessage::kType, nil_size, 0);
        serde::Write(s, NilMessage { .size = nil_size });
    }

    static Object AllocateEmptyAtEOF(len_t min_size, const std::shared_ptr<FileLink>& file);

public:
    std::shared_ptr<FileLink> file;

private:
    struct Space {
        offset_t offset;
        len_t size;
    };

    [[nodiscard]] cstd::optional<Space> FindSpace(size_t size, bool must_be_nil) const;

    // TODO: refactor to not be recursive
    // TODO: take predicate visitor?
    template<serde::Deserializer D>
    [[nodiscard]] static cstd::optional<Object::Space> FindMessageRecursive(  // NOLINT(*-no-recursion
        D& de,
        offset_t sb_base_addr,
        uint16_t& messages_read,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint16_t msg_type
    ) {
        uint32_t bytes_read = 0;

        while (bytes_read < size_limit && messages_read < total_message_ct) {
            auto type = serde::Read<D, uint16_t>(de);
            auto size_bytes = serde::Read<D, uint16_t>(de);

            bytes_read += size_bytes + kPrefixSize;
            ++messages_read;

            // flags + reserved
            serde::Skip(de, 4);

            if (type == ObjectHeaderContinuationMessage::kType) {
                auto cont = serde::Read<D, ObjectHeaderContinuationMessage>(de);

                offset_t return_pos = de.GetPosition();
                de.SetPosition(sb_base_addr + cont.offset);

                cstd::optional<Space> found = FindMessageRecursive(de, sb_base_addr, messages_read, total_message_ct, cont.length, msg_type);

                if (found.has_value()) {
                    return found;
                }

                de.SetPosition(return_pos);
            } else {
                if (type == msg_type) {
                    uint16_t total_size = size_bytes + kPrefixSize;

                    return Space {
                        .offset = de.GetPosition() - kPrefixSize,
                        .size = total_size,
                    };
                }

                for (size_t b = 0; b < size_bytes; ++b) {
                    serde::Skip<D, byte_t>(de);
                }
            }
        }

        return cstd::nullopt;
    }

    // TODO: refactor to not be recursive
    template<serde::Deserializer D>
    [[nodiscard]] static cstd::optional<Object::Space> FindSpaceRecursive(  // NOLINT(*-no-recursion
        D& de,
        offset_t sb_base_addr,
        uint16_t& messages_read,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint32_t search_size,
        bool must_be_nil
    ) {
        uint32_t bytes_read = 0;

        cstd::optional<Space> smallest_found{};

        while (bytes_read < size_limit && messages_read < total_message_ct) {
            auto type = serde::Read<D, uint16_t>(de);
            auto size_bytes = serde::Read<D, uint16_t>(de);

            bytes_read += size_bytes + kPrefixSize;
            ++messages_read;

            // flags + reserved
            serde::Skip(de, 4);

            if (type == ObjectHeaderContinuationMessage::kType) {
                auto cont = serde::Read<D, ObjectHeaderContinuationMessage>(de);

                offset_t return_pos = de.GetPosition();
                de.SetPosition(sb_base_addr + cont.offset);

                cstd::optional<Space> res = FindSpaceRecursive(de, sb_base_addr, messages_read, total_message_ct, cont.length, search_size, must_be_nil);

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
                    serde::Skip(de, 4);
                }
            }
        }

        return smallest_found;
    }

    void JumpToRelativeOffset(offset_t offset) const {
        file->io.SetPosition(file_pos_ + offset);
    }

    explicit Object(const std::shared_ptr<FileLink>& file, offset_t pos_)
        : file(file), file_pos_(pos_) {}

private:
    offset_t file_pos_{};
};
