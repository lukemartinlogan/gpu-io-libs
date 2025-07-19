#include "tree.h"

#include <stdexcept>

void BTreeChunkedRawDataNodeKey::Serialize(Serializer& s) const {
    s.Write(chunk_size);

    for (const uint64_t offset: chunk_offset_in_dataset) {
        s.Write(offset);
    }

    s.Write<uint64_t>(0);
}

BTreeChunkedRawDataNodeKey BTreeChunkedRawDataNodeKey::Deserialize(Deserializer& de) {
    BTreeChunkedRawDataNodeKey key{};

    key.chunk_size = de.Read<uint32_t>();

    for (uint64_t offset; (offset = de.Read<uint64_t>()) != 0;) {
        key.chunk_offset_in_dataset.push_back(offset);
    }

    return key;
}

template<typename K>
void WriteEntries(const BTreeEntries<K>& entries, Serializer& s) {
    uint16_t entries_ct = entries.child_pointers.size();

    if (entries.keys.size() != entries_ct + 1) {
        throw std::logic_error("Shape of entries was invalid");
    }

    for (uint16_t i = 0; i < entries_ct; ++i) {
        s.Write(entries.keys.at(i));
        s.Write(entries.child_pointers.at(i));
    }

    s.Write(entries.keys.back());
}

void BTreeNode::Serialize(Serializer& s) const {
    uint8_t type;
    if (std::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        type = kGroupNodeTy;
    } else if (std::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        type = kRawDataChunkNodeTy;
    } else {
        throw std::logic_error("Variant has invalid state");
    }

    s.Write(kTreeSignature);

    s.Write(type);
    s.Write(level);
    s.Write(entries_used);

    s.Write(left_sibling_addr);
    s.Write(right_sibling_addr);

    if (type == kGroupNodeTy) {
        const auto& entr = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);
        WriteEntries(entr, s);
    } else {
        const auto& entr = std::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);
        WriteEntries(entr, s);
    }
}

template<typename K>
BTreeEntries<K> ReadEntries(uint16_t entries_used, Deserializer& de) {
    BTreeEntries<K> entries{};

    for (uint16_t i = 0; i < entries_used; ++i) {
        entries.keys.push_back(de.ReadComplex<K>());
        entries.child_pointers.push_back(de.Read<offset_t>());
    }

    entries.keys.push_back(de.ReadComplex<K>());

    return entries;
}

BTreeNode BTreeNode::Deserialize(Deserializer& de) {
    if (de.Read<std::array<uint8_t, 4>>() != kTreeSignature) {
        throw std::runtime_error("BTree signature was invalid");
    }

    auto type = de.Read<uint8_t>();

    if (type != kGroupNodeTy && type != kRawDataChunkNodeTy) {
        throw std::runtime_error("Invalid BTree node type");
    }

    BTreeNode node{};

    node.level = de.Read<uint8_t>();
    node.entries_used = de.Read<uint16_t>();

    node.left_sibling_addr = de.Read<offset_t>();
    node.right_sibling_addr = de.Read<offset_t>();

    if (type == kGroupNodeTy) {
        node.entries = ReadEntries<BTreeGroupNodeKey>(node.entries_used, de);
    } else /* kRawDataChunkNodeTy */ {
        node.entries = ReadEntries<BTreeChunkedRawDataNodeKey>(node.entries_used, de);
    }

    return node;
}
