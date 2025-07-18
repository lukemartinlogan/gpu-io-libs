#pragma once
#include <cstdint>
#include <variant>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

struct SymbolTableMessage {
    // address of v1 b-tree containing symbol table entries
    offset_t b_tree_addr;
    // address of local heap containing link names
    // for symbol table entries
    offset_t local_heap_addr;

    void Serialize(Serializer& s) const {
        s.WriteRaw(*this);
    }

    static SymbolTableMessage Deserialize(Deserializer& de) {
        return de.ReadRaw<SymbolTableMessage>();
    }

    static constexpr uint16_t kType = 0x11;
};

struct ObjectHeaderMessage {
    // TODO: this can be stored in the variant
    enum class Type : uint16_t {
        // ignore message, variable length
        kNil = 0x0000,
        // exactly 1 req for datasets
        // variable len based on num of dimensions
        kDataspace = 0x0001,
        // ?current state of links
        kLinkInfo = 0x0002,
        // exactly 1 req for datasets
        // datatype for each elem of dataset
        kDatatype = 0x0003,
        // uninit value
        kFillValueOld = 0x0004,
        // uninit value, same datatype as dataset
        kFillValue = 0x0005,
        // info for link in group object header
        kLink = 0x0006,
        // indicated data for object stored out of file
        kExternalDataFiles = 0x0007,
        // how elems of multi dimensions array are stored
        kDataLayout = 0x0008,
        // for testing, should never appear
        kBogus = 0x0009,
        // info for constants defining group behavior
        kGroupInfo = 0x000a,
        //
        kFilterPipeline = 0x000b,
        //
        kAttribute = 0x000c,
        //
        kObjectComment = 0x000d,
        //
        kObjectModificationTimeOld = 0x000e,
        //
        kSharedMessageTable = 0x000f,
        //
        kObjectHeaderContinuation = 0x0010,
        //
        kSymbolTable = 0x0011,
        //
        kObjectModificationTime = 0x0012,
        //
        kBTreeKValues = 0x0013,
        //
        kDriverInfo = 0x0014,
        //
        kAttributeInfo = 0x0015,
        //
        kObjectRefCount = 0x0016,
        //
        kFileSpaceInfo = 0x0017,
    } type;

    // if this gets too large, put it on the heap
    std::variant<SymbolTableMessage, std::monostate> message{};
    uint8_t flags;

    uint16_t Size() const {
        uint16_t size{};

        switch (type) {
            case Type::kSymbolTable: {
                size = sizeof(SymbolTableMessage);
            }
            default: {

            }
        }

        return size + 8 * sizeof(byte_t);
    }

    void Serialize(Serializer& s) const;

    static ObjectHeaderMessage Deserialize(Deserializer& de);
};

struct ObjectHeader {
    // total number of messages listed
    // includes continuation messages
    uint16_t message_count{};
    // number of hard links to this object in the current file
    uint32_t object_ref_count{};
    // number of bytes of header message data for this header
    // does not include size of object header continuation blocks
    uint32_t object_header_size{};
    // messages
    std::vector<ObjectHeaderMessage> messages{};

    void Serialize(Serializer& s) const;

    static ObjectHeader Deserialize(Deserializer& de);
private:
    static constexpr uint8_t kVersionNumber = 0x01;
};
