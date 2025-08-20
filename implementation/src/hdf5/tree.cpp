#include <stdexcept>

#include "tree.h"
#include "local_heap.h"

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

template <typename K>
uint16_t BTreeEntries<K>::EntriesUsed() const {
    uint16_t entries_ct = child_pointers.size();

    if (keys.size() != entries_ct + 1) {
        throw std::logic_error("Shape of entries was invalid");
    }

    return entries_ct;
}

uint16_t BTreeNode::EntriesUsed() const {
    return std::visit([](const auto& entries) {
        return entries.EntriesUsed();
    }, entries);
}

std::optional<offset_t> BTreeNode::Get(std::string_view name, FileLink& file, const LocalHeap& heap) const { // NOLINT(*-no-recursion
    std::optional<uint16_t> child_index = FindIndex(name, heap, file.io);

    if (!child_index) {
        return std::nullopt;
    }

    const auto& group_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    // leaf node, search for the exact entry
    // pointers point to symbol table entries
    if (level == 0) {
        return group_entries.child_pointers.at(*child_index);
    }

    // recursively search the tree
    offset_t child_addr = group_entries.child_pointers.at(*child_index);

    file.io.SetPosition(child_addr);
    auto child_node = file.io.ReadComplex<BTreeNode>();

    return child_node.Get(name, file, heap);
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

std::optional<uint16_t> BTreeNode::FindIndex(std::string_view key, const LocalHeap& heap, Deserializer& de) const {
    if (!std::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        return std::nullopt;
    }

    const auto& group_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        // empty node, no entries
        return std::nullopt;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct + 1;

    std::string prev = heap.ReadString(group_entries.keys.front().first_object_name, de);

    for (size_t i = 1; i < group_entries.keys.size(); ++i) {
        std::string next = heap.ReadString(group_entries.keys[i].first_object_name, de);

        if (prev < key && key <= next) {
            child_index = i - 1;
            break;
        }

        prev = std::move(next);
    }

    if (child_index == entries_ct + 1) {
        // name is greater than all keys
        return std::nullopt;
    }

    return child_index;
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

    s.Write(kSignature);

    s.Write(type);
    s.Write(level);
    s.Write(EntriesUsed());

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
    if (de.Read<std::array<uint8_t, 4>>() != kSignature) {
        throw std::runtime_error("BTree signature was invalid");
    }

    auto type = de.Read<uint8_t>();

    if (type != kGroupNodeTy && type != kRawDataChunkNodeTy) {
        throw std::runtime_error("Invalid BTree node type");
    }

    BTreeNode node{};

    node.level = de.Read<uint8_t>();
    auto entries_used = de.Read<uint16_t>();

    node.left_sibling_addr = de.Read<offset_t>();
    node.right_sibling_addr = de.Read<offset_t>();

    if (type == kGroupNodeTy) {
        node.entries = ReadEntries<BTreeGroupNodeKey>(entries_used, de);
    } else /* kRawDataChunkNodeTy */ {
        node.entries = ReadEntries<BTreeChunkedRawDataNodeKey>(entries_used, de);
    }

    return node;
}
