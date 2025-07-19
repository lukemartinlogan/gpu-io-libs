#pragma once
#include <array>
#include <variant>
#include <vector>

#include "types.h"

inline constexpr std::array<uint8_t, 4> kTreeSignature = { 0x54, 0x52, 0x45, 0x45 };

struct BTreeGroupNodeKey {
    // byte offset into local heap
    // first object name in the subtree the key describes
    len_t first_object_name;
};

struct BTreeChunkedRawDataNodeKey {
    // in bytes
    uint32_t chunk_size;
    // .size() == number of dimensions
    // extra uint64_t(0) at the end (not stored)
    std::vector<uint64_t> chunk_offset_in_dataset;
};

template<typename K>
struct BTreeEntry {
    K key;
    offset_t child_pointer;
};

using BTreeEntries = std::variant<std::vector<BTreeEntry<BTreeGroupNodeKey>>, std::vector<BTreeEntry<BTreeChunkedRawDataNodeKey>>>;

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
    BTreeEntries entries;
};