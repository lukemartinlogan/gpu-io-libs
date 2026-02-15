#pragma once

#include "file_link.h"
#include "types.h"
#include "object_header.h"

// TODO: create iterator over messages
class Object {
public:
    // FIXME: get rid of this ctor
    __device__
    Object() = default;

    __device__
    static hdf5::expected<Object> New(FileLink* file, offset_t pos_) {
        auto io = file->MakeRW();
        io.SetPosition(pos_);

        // FIXME: hardcoded constant
        if (serde::Read<uint8_t>(io) != 0x01) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidVersion, "Object version number was invalid");
        }

        return Object(file, pos_);
    }

    __device__
    [[nodiscard]] hdf5::expected<ObjectHeader> GetHeader() const {
        auto io = file->MakeRW();
        io.SetPosition(file_pos_);
        return serde::Read<ObjectHeader>(io);
    }

    __device__
    [[nodiscard]] offset_t GetAddress() const {
        return file_pos_;
    }

    // TODO: should this mutate an internally held object as well?
    // TODO: add a 'dirty' field to header messages
    __device__
    void WriteMessage(const HeaderMessageVariant& msg) const;

    __device__
    cstd::optional<ObjectHeaderMessage> DeleteMessage(uint16_t msg_type);

    template<typename T>
    __device__ cstd::optional<T> DeleteMessage();

    __device__
    cstd::optional<ObjectHeaderMessage> GetMessage(uint16_t msg_type);

    template<typename T>
    __device__ cstd::optional<T> GetMessage();

    template<serde::Serializer S>
    __device__
    static void WriteEmpty(len_t min_size, S& s);

    __device__
    static Object AllocateEmptyAtEOF(len_t min_size, FileLink* file);

public:
    FileLink* file;

private:
    struct Space {
        offset_t offset;
        len_t size;
    };

    __device__
    [[nodiscard]] cstd::optional<Space> FindSpace(size_t size, bool must_be_nil) const;

    template<serde::Deserializer D>
    __device__
    [[nodiscard]] static cstd::optional<Space> FindMessage(
        D& de,
        offset_t sb_base_addr,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint16_t msg_type
    );

    template<serde::Deserializer D>
    __device__
    [[nodiscard]] static cstd::optional<Object::Space> FindSpace(
        D& de,
        offset_t sb_base_addr,
        uint16_t total_message_ct,
        uint32_t size_limit,
        uint32_t search_size,
        bool must_be_nil
    );

    __device__
    explicit Object(FileLink* file, offset_t pos_)
        : file(file), file_pos_(pos_) {}

private:
    offset_t file_pos_{};
};

template<typename T>
__device__ inline cstd::optional<T> Object::DeleteMessage() {
    cstd::optional<ObjectHeaderMessage> msg = DeleteMessage(T::kType);
    if (msg.has_value()) {
        return cstd::get<T>(msg->message);
    } else {
        return cstd::nullopt;
    }
}

template<typename T>
__device__ inline cstd::optional<T> Object::GetMessage() {
    cstd::optional<ObjectHeaderMessage> msg = GetMessage(T::kType);
    if (msg.has_value()) {
        return cstd::get<T>(msg->message);
    } else {
        return cstd::nullopt;
    }
}

template<serde::Serializer S>
__device__ inline void Object::WriteEmpty(len_t min_size, S& s) {
    // TODO: this probably shouldn't be unused!
    len_t aligned_size = EmptyHeaderMessagesSize(min_size);

    serde::Write(s, static_cast<uint8_t>(0x01));

    // reserved
    serde::Write<uint8_t>(s, 0);
    // total num of messages (one nil message)
    serde::Write<uint16_t>(s, 1);

    // object ref count
    serde::Write<uint32_t>(s, 0);
    // header size
    serde::Write<uint32_t>(s, min_size);

    // reserved
    serde::Write<uint32_t>(s, 0);

    // TODO: fix size overflow?
    uint16_t nil_size = min_size - 8;

    WriteHeader(s, NilMessage::kType, nil_size, 0);
    serde::Write(s, NilMessage { nil_size });
}

template<serde::Deserializer D>
__device__ inline cstd::optional<Object::Space> Object::FindMessage(
    D& de,
    offset_t sb_base_addr,
    uint16_t total_message_ct,
    uint32_t size_limit,
    uint16_t msg_type
) {
    struct StackFrame {
        offset_t return_pos;
        uint32_t size_limit;
        uint32_t bytes_read;
    };

    cstd::inplace_vector<StackFrame, ObjectHeader::kMaxContinuationDepth> stack;
    stack.push_back({de.GetPosition(), size_limit, 0});

    uint16_t messages_read = 0;

    while (!stack.empty()) {
        auto& frame = stack.back();

        if (frame.bytes_read >= frame.size_limit || messages_read >= total_message_ct) {
            // Only restore position for continuation frames, not the initial frame
            if (stack.size() > 1) {
                de.SetPosition(frame.return_pos);
            }
            stack.pop_back();
            continue;
        }

        auto type = serde::Read<uint16_t>(de);
        auto size_bytes = serde::Read<uint16_t>(de);

        frame.bytes_read += size_bytes + kPrefixSize;
        ++messages_read;

        // flags + reserved
        serde::Skip(de, 4);

        if (type == ObjectHeaderContinuationMessage::kType) {
            auto cont = serde::Read<ObjectHeaderContinuationMessage>(de);

            offset_t return_pos = de.GetPosition();
            de.SetPosition(sb_base_addr + cont.offset);

            stack.push_back({return_pos, static_cast<uint32_t>(cont.length), 0});
        } else {
            if (type == msg_type) {
                uint16_t total_size = size_bytes + kPrefixSize;

                return Space { de.GetPosition() - kPrefixSize, total_size };
            }

            for (size_t b = 0; b < size_bytes; ++b) {
                serde::Skip<byte_t>(de);
            }
        }
    }

    return cstd::nullopt;
}

template<serde::Deserializer D>
__device__ inline cstd::optional<Object::Space> Object::FindSpace(
    D& de,
    offset_t sb_base_addr,
    uint16_t total_message_ct,
    uint32_t size_limit,
    uint32_t search_size,
    bool must_be_nil
) {
    struct StackFrame {
        offset_t return_pos;
        uint32_t size_limit;
        uint32_t bytes_read;
    };

    cstd::inplace_vector<StackFrame, ObjectHeader::kMaxContinuationDepth> stack;
    stack.push_back({de.GetPosition(), size_limit, 0});

    uint16_t messages_read = 0;
    cstd::optional<Space> smallest_found{};

    while (!stack.empty()) {
        auto& frame = stack.back();

        if (frame.bytes_read >= frame.size_limit || messages_read >= total_message_ct) {
            // Only restore position for continuation frames, not the initial frame
            if (stack.size() > 1) {
                de.SetPosition(frame.return_pos);
            }
            stack.pop_back();
            continue;
        }

        auto type = serde::Read<uint16_t>(de);
        auto size_bytes = serde::Read<uint16_t>(de);

        frame.bytes_read += size_bytes + kPrefixSize;
        ++messages_read;

        // flags + reserved
        serde::Skip(de, 4);

        if (type == ObjectHeaderContinuationMessage::kType) {
            auto cont = serde::Read<ObjectHeaderContinuationMessage>(de);

            offset_t return_pos = de.GetPosition();
            de.SetPosition(sb_base_addr + cont.offset);

            stack.push_back({return_pos, static_cast<uint32_t>(cont.length), 0});
        } else {
            if (!must_be_nil || type == NilMessage::kType) {
                uint16_t total_size = size_bytes + kPrefixSize;

                if (
                    // no new nil header needed || nil header needed
                    total_size == search_size || total_size >= search_size + kPrefixSize
                    && ( !smallest_found.has_value() || total_size < smallest_found->size )
                ) {
                    smallest_found = Space { de.GetPosition() - kPrefixSize, total_size };
                }
            }

            for (size_t b = 0; b < size_bytes; ++b) {
                serde::Skip(de, 4);
            }
        }
    }

    return smallest_found;
}
