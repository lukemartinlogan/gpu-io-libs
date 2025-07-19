#pragma once
#include <array>
#include <variant>
#include <vector>

#include "types.h"
#include "../serialization/serialization.h"

struct BTreeGroupNodeKey {
    // byte offset into local heap
    // first object name in the subtree the key describes
    len_t first_object_name;

    void Serialize(Serializer& s) const {
        s.WriteRaw<BTreeGroupNodeKey>(*this);
    }

    static BTreeGroupNodeKey Deserialize(Deserializer& de) {
        return de.ReadRaw<BTreeGroupNodeKey>();
    }
};

struct BTreeChunkedRawDataNodeKey {
    // in bytes
    uint32_t chunk_size;
    // .size() == number of dimensions
    // extra uint64_t(0) at the end (not stored)
    std::vector<uint64_t> chunk_offset_in_dataset;

    void Serialize(Serializer& s) const;

    static BTreeChunkedRawDataNodeKey Deserialize(Deserializer& de);
};

template<typename K>
struct BTreeEntries {
    // TODO: enforce child_pointers.size() + 1 == keys.size()
    std::vector<K> keys;
    std::vector<offset_t> child_pointers;
};

struct BTreeNode {
    // type: (not stored, check variant)
    // implies max degree K of the tree & size of each key field

    // what level node appears in the tree, leaf nodes are at zero
    // indicates if child pointers point to subtrees or to data
    uint8_t level{};
    // max number of children this node points to
    // all nodes have same max degree (max entries used) but
    // most nodes point to less than that
    uint16_t entries_used{};
    // relative addr of curr node's left sibling
    // if leftmost, then kUndefinedOffset
    offset_t left_sibling_addr = kUndefinedOffset;
    // relative addr of curr node's right sibling
    // if rightmost, then kUndefinedOffset
    offset_t right_sibling_addr = kUndefinedOffset;
    // last entry's child_pointer field is unused
    std::variant<
        BTreeEntries<BTreeGroupNodeKey>,
        BTreeEntries<BTreeChunkedRawDataNodeKey>
    > entries{};

private:
    static constexpr std::array<uint8_t, 4> kTreeSignature = { 0x54, 0x52, 0x45, 0x45 };
};