#include <stdexcept>

#include "tree.h"
#include "local_heap.h"

void BTreeChunkedRawDataNodeKey::Serialize(Serializer& s) const {
    s.Write(chunk_size);
    s.Write(filter_mask);

    for (const uint64_t offset: chunk_offset_in_dataset.coords) {
        s.Write(offset);
    }

    s.Write<uint64_t>(0);
}

BTreeChunkedRawDataNodeKey BTreeChunkedRawDataNodeKey::DeserializeWithDims(Deserializer& de, uint8_t dimensionality) {
    BTreeChunkedRawDataNodeKey key{};

    key.chunk_size = de.Read<uint32_t>();
    key.filter_mask = de.Read<uint32_t>();

    for (uint8_t i = 0; i < dimensionality; ++i) {
        auto offset = de.Read<uint64_t>();

        key.chunk_offset_in_dataset.coords.push_back(offset);

        if (offset == 0) {
            throw std::runtime_error("BTreeChunkedRawDataNodeKey: unexpected 0 in chunk coordinates");
        }
    }

    if (de.Read<uint64_t>() != 0) {
        throw std::runtime_error("BTreeChunkedRawDataNodeKey: expected terminating 0");
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

template <typename K>
uint16_t BTreeEntries<K>::KeySize() const {
    if constexpr (std::is_same_v<K, BTreeGroupNodeKey>) {
        return BTreeGroupNodeKey::kAllocationSize;
    } else if constexpr (std::is_same_v<K, BTreeChunkedRawDataNodeKey>) {
        if (keys.empty()) {
            throw std::logic_error("Cannot determine key size for empty entries");
        }

        return keys.front().AllocationSize();
    } else {
        throw std::logic_error("unsupported key type");
    }
}

uint16_t BTreeNode::EntriesUsed() const {
    return std::visit([](const auto& entries) {
        return entries.EntriesUsed();
    }, entries);
}

std::optional<offset_t> BTreeNode::Get(std::string_view name, FileLink& file, const LocalHeap& heap) const { // NOLINT(*-no-recursion
    std::optional<uint16_t> child_index = FindGroupIndex(name, heap, file.io);

    if (!child_index) {
        return std::nullopt;
    }

    const auto& group_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    // leaf node, search for the exact entry
    // pointers point to symbol table entries
    if (IsLeaf()) {
        return group_entries.child_pointers.at(*child_index);
    }

    // recursively search the tree
    offset_t child_addr = group_entries.child_pointers.at(*child_index);

    file.io.SetPosition(file.superblock.base_addr + child_addr);
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

std::optional<uint16_t> BTreeNode::FindGroupIndex(std::string_view key, const LocalHeap& heap, Deserializer& de) const {
    if (!std::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        return std::nullopt;
    }

    // TODO: can we binary search here?

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

uint16_t BTreeNode::GroupInsertionPosition(std::string_view key, const LocalHeap& heap, Deserializer& de) const {
    if (!std::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        throw std::logic_error("InsertionPosition only supported for group nodes");
    }

    // TODO: can we binary search here?

    const auto& group_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        return 0;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct;

    for (size_t i = 0; i < entries_ct; ++i) {
        std::string next = heap.ReadString(group_entries.keys.at(i + 1).first_object_name, de);

        if (key <= next) {
            child_index = i;
            break;
        }
    }

    return child_index;
}

std::optional<uint16_t> BTreeNode::FindChunkedIndex(const ChunkCoordinates& chunk_coords) const {
    if (!std::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        return std::nullopt;
    }

    const auto& chunk_entries = std::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        // empty node, no entries
        return std::nullopt;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct + 1;

    ChunkCoordinates prev = chunk_entries.keys.front().chunk_offset_in_dataset;

    for (size_t i = 1; i < chunk_entries.keys.size(); ++i) {
        ChunkCoordinates next = chunk_entries.keys[i].chunk_offset_in_dataset;

        if (prev <= chunk_coords && chunk_coords < next) {
            child_index = i - 1;
            break;
        }

        prev = std::move(next);
    }

    if (child_index == entries_ct + 1) {
        // coords are greater than all keys
        return std::nullopt;
    }

    return child_index;
}

uint16_t BTreeNode::ChunkedInsertionPosition(const ChunkCoordinates& chunk_coords) const {
    if (!std::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        throw std::logic_error("ChunkedInsertionPosition only supported for chunked nodes");
    }

    const auto& chunk_entries = std::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        return 0;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct;

    for (size_t i = 0; i < entries_ct; ++i) {
        const auto& next = chunk_entries.keys.at(i + 1).chunk_offset_in_dataset;

        if (chunk_coords < next) {
            child_index = i;
            break;
        }
    }

    return child_index;
}

template<typename K>
K BTreeNode::GetMaxKey(FileLink& file) const {
    static_assert(
        std::is_same_v<K, BTreeGroupNodeKey> || std::is_same_v<K, BTreeChunkedRawDataNodeKey>,
        "Unsupported key type"
    );

    if (!std::holds_alternative<BTreeEntries<K>>(entries)) {
        throw std::logic_error("GetMaxKey: incorrect key type for this node");
    }

    auto node_entries = std::get<BTreeEntries<K>>(entries);

    if (node_entries.EntriesUsed() == 0) {
        throw std::logic_error("GetMaxKey called on empty node");
    }

    if (IsLeaf()) {
        return node_entries.keys.back();
    } else {
        file.io.SetPosition(file.superblock.base_addr + node_entries.child_pointers.back());
        auto child = file.io.ReadComplex<BTreeNode>();

        return child.GetMaxKey<K>(file);
    }
}

template<typename K>
K BTreeNode::GetMinKey() const {
    static_assert(
        std::is_same_v<K, BTreeGroupNodeKey> || std::is_same_v<K, BTreeChunkedRawDataNodeKey>,
        "Unsupported key type"
    );

    if (!std::holds_alternative<BTreeEntries<K>>(entries)) {
        throw std::logic_error("GetMinKey: incorrect key type for this node");
    }

    auto node_entries = std::get<BTreeEntries<K>>(entries);

    if (node_entries.EntriesUsed() == 0) {
        throw std::logic_error("GetMinKey called on empty node");
    }

    return node_entries.keys.front();
}

len_t BTreeNode::AllocationSize(KValues k_val) const {
    const uint16_t k = k_val.Get(IsLeaf());

    uint16_t key_size = std::visit([](const auto& entries) -> uint16_t { return entries.KeySize(); }, entries);

    return
        + 4 // Signature

        + 1 // Node Type
        + 1 // Node Level
        + 2 // Entries Used

        + sizeof(offset_t) // Left Sibling Address
        + sizeof(offset_t) // Right Sibling Address

        + (2 * k + 1) * key_size // keys
        + (2 * k) * sizeof(offset_t) // child pointers
    ;
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

bool BTreeNode::AtCapacity(KValues k) const {
    return EntriesUsed() == k.Get(IsLeaf()) * 2;
}

BTreeNode BTreeNode::Split(KValues k) {
    return std::visit([this, k]<typename Entries>(Entries& l_entries) -> BTreeNode {
        Entries r_entries{};
        uint16_t mid = k.Get(IsLeaf());

        // 1. move keys to right node
        r_entries.keys.assign(l_entries.keys.begin() + mid, l_entries.keys.end());
        // + 1 to keep key
        l_entries.keys.erase(l_entries.keys.begin() + mid + 1, l_entries.keys.end());

        // 2. move pointers
        r_entries.child_pointers.assign(l_entries.child_pointers.begin() + mid, l_entries.child_pointers.end());
        l_entries.child_pointers.erase(l_entries.child_pointers.begin() + mid, l_entries.child_pointers.end());

        return {
            .level = level,
            .entries = r_entries,
        };
    }, entries);
}

len_t BTreeNode::WriteNodeGetAllocSize(offset_t offset, FileLink& file, KValues k) const {
    file.io.SetPosition(offset);
    file.io.WriteComplex(*this);

    len_t written_bytes = file.io.GetPosition() - offset;

    uint16_t max_entries = 2 * k.Get(IsLeaf());

    if (EntriesUsed() > max_entries) {
        throw std::logic_error("AllocateWriteNewNode called on over-capacity node");
    }

    len_t unused_entries = 2 * k.Get(IsLeaf()) - EntriesUsed();

    // Calculate key size based on node type
    uint16_t key_size = std::visit([](const auto& entries) -> uint16_t { return entries.KeySize(); }, entries);

    len_t key_ptr_size = key_size + sizeof(offset_t);

    // intended allocation size
    return written_bytes + unused_entries * key_ptr_size;
};

offset_t BTreeNode::AllocateAndWrite(FileLink& file, KValues k) const {
    len_t alloc_size = AllocationSize(k);
    offset_t alloc_start = file.AllocateAtEOF(alloc_size);

    len_t intended_alloc_size = WriteNodeGetAllocSize(alloc_start, file, k);

    if (alloc_size != intended_alloc_size) {
        throw std::logic_error("AllocateWriteNewNode: size mismatch");
    }

    return alloc_start;
}

void BTreeNode::Recurse(const std::function<void(std::string, offset_t)>& visitor, FileLink& file) const {
    if (!std::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        throw std::logic_error("Recurse only supported for group nodes");
    }

    auto g_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    for (size_t i = 0; i < g_entries.EntriesUsed(); ++i) {
        offset_t ptr = g_entries.child_pointers.at(i);

        if (IsLeaf()) {
            std::string name = std::to_string(g_entries.keys.at(i).first_object_name);

            visitor(std::move(name), ptr);
        } else {
            file.io.SetPosition(file.superblock.base_addr + ptr);
            auto child = file.io.ReadComplex<BTreeNode>();

            child.Recurse(visitor, file);
        }
    }
}

std::optional<SplitResult> BTreeNode::InsertGroup(offset_t this_offset, offset_t name_offset, offset_t obj_header_ptr, FileLink& file, LocalHeap& heap) {
    std::optional<SplitResult> res{};

    std::string name_str = heap.ReadString(name_offset, file.io);

    const KValues k {
        .leaf = file.superblock.group_leaf_node_k,
        .internal = file.superblock.group_internal_node_k,
    };

    auto& g_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    auto RawInsert = [&file, &heap](BTreeNode& node, BTreeGroupNodeKey key, offset_t child_ptr) -> void {
        std::string key_str = heap.ReadString(key.first_object_name, file.io);

        auto& ins_entries = std::get<BTreeEntries<BTreeGroupNodeKey>>(node.entries);
        uint16_t ins_pos = node.GroupInsertionPosition(key_str, heap, file.io);

        ins_entries.child_pointers.insert(
            ins_entries.child_pointers.begin() + ins_pos,
            child_ptr
        );

        ins_entries.keys.insert( // keys are offset by one
            ins_entries.keys.begin() + ins_pos + 1,
            key
        );
    };

    if (IsLeaf()) {
        if (AtCapacity(k)) {
            // do we alloc a new string?
            uint16_t mid_index = k.leaf;

            BTreeGroupNodeKey promoted_key = g_entries.keys.at(mid_index);
            BTreeNode new_node = Split(k);

            std::string promoted_key_str = heap.ReadString(promoted_key.first_object_name, file.io);

            // TODO: is < or <= ?
            if (name_str <= promoted_key_str) {
                RawInsert(*this, { name_offset }, obj_header_ptr);
            } else {
                RawInsert(new_node, { name_offset }, obj_header_ptr);
            }

            offset_t new_node_alloc = new_node.AllocateAndWrite(file, k);

            res = {
                .promoted_key = promoted_key,
                .new_node_offset = new_node_alloc,
            };
        } else {
            RawInsert(*this, { name_offset }, obj_header_ptr);
        }

        WriteNodeGetAllocSize(this_offset, file, k);
    } else {
        std::optional<uint16_t> child_idx = FindGroupIndex(name_str, heap, file.io);

        if (!child_idx) {
            throw std::runtime_error("BTreeNode::Insert: could not find child index");
        }

        offset_t child_offset = g_entries.child_pointers.at(*child_idx);

        file.io.SetPosition(child_offset);
        auto child = file.io.ReadComplex<BTreeNode>();

        std::optional<SplitResult> child_ins = child.InsertGroup(child_offset, name_offset, obj_header_ptr, file, heap);

        if (child_ins.has_value()) {
            if (AtCapacity(k)) {
                uint16_t mid_index = k.internal;

                BTreeGroupNodeKey promoted_key = g_entries.keys.at(mid_index);
                BTreeNode new_node = Split(k);

                std::string promoted_key_str = heap.ReadString(promoted_key.first_object_name, file.io);

                // TODO: is < or <= ?
                if (name_str <= promoted_key_str) {
                    RawInsert(*this, child_ins->promoted_key, child_ins->new_node_offset);
                } else {
                    RawInsert(new_node, child_ins->promoted_key, child_ins->new_node_offset);
                }

                offset_t new_node_alloc = new_node.AllocateAndWrite(file, k);

                res = {
                    .promoted_key = promoted_key,
                    .new_node_offset = new_node_alloc,
                };
            } else {
                RawInsert(*this, child_ins->promoted_key, child_ins->new_node_offset);
            }

            WriteNodeGetAllocSize(this_offset, file, k);
        }
    }

    return res;
}

std::optional<offset_t> GroupBTree::Get(std::string_view name) const {
    std::optional<BTreeNode> root = ReadRoot();

    if (!root.has_value()) {
        return std::nullopt;
    }

    return root->Get(name, *file_, heap_);
}

void GroupBTree::InsertGroup(offset_t name_offset, offset_t object_header_ptr) {
    const BTreeNode::KValues k {
        .leaf = file_->superblock.group_leaf_node_k,
        .internal = file_->superblock.group_internal_node_k
    };

    std::optional<BTreeNode> root = ReadRoot();

    if (!root.has_value()) {
        BTreeEntries<BTreeGroupNodeKey> entries{};

        offset_t empty_str_offset = heap_.WriteString("", *file_);

        entries.keys.push_back({ empty_str_offset });
        entries.child_pointers.push_back(/* root: */ object_header_ptr);
        entries.keys.push_back({ name_offset });

        BTreeNode new_root {
            .level = 0,
            .entries = entries,
        };

        addr_ = new_root.AllocateAndWrite(*file_, k);

        return;
    }

    std::optional<SplitResult> split = root->InsertGroup(*addr_, name_offset, object_header_ptr, *file_, heap_);

    if (split.has_value()) {
        BTreeEntries<BTreeGroupNodeKey> entries{};

        auto min = root->GetMinKey<BTreeGroupNodeKey>(), max = root->GetMaxKey<BTreeGroupNodeKey>(*file_);

        entries.keys.push_back(min);
        entries.child_pointers.push_back(/* root: */ *addr_);
        entries.keys.push_back(split->promoted_key);
        entries.child_pointers.push_back(split->new_node_offset);
        entries.keys.push_back(max);

        if (root->level == std::numeric_limits<uint8_t>::max()) {
            throw std::runtime_error("BTree level overflow");
        }

        BTreeNode new_root {
            .level = static_cast<uint8_t>(root->level + 1),
            .entries = entries,
        };

        addr_ = new_root.AllocateAndWrite(*file_, k);
    }
}

size_t GroupBTree::Size() const {
    std::optional<BTreeNode> root = ReadRoot();

    if (!root.has_value()) {
        return 0;
    }

    size_t size = 0;

    root->Recurse([&size](const std::string&, offset_t) { ++size; }, *file_);

    return size;
}

std::vector<offset_t> GroupBTree::Elements() const {
    std::optional<BTreeNode> root = ReadRoot();

    if (!root.has_value()) {
        return {};
    }

    std::vector<offset_t> elems;

    root->Recurse([&elems](const std::string&, offset_t ptr) { elems.push_back(ptr); }, *file_);

    return elems;
}

std::optional<BTreeNode> GroupBTree::ReadRoot() const {
    if (!addr_.has_value()) {
        return std::nullopt;
    }

    file_->io.SetPosition(*addr_);

    return file_->io.ReadComplex<BTreeNode>();
}
