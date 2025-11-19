#include <stdexcept>

#include "types.h"
#include "tree.h"
#include "local_heap.h"

void BTreeChunkedRawDataNodeKey::Serialize(Serializer& s) const {
    s.Write(chunk_size);
    s.Write(filter_mask);

    for (const uint64_t offset: chunk_offset_in_dataset.coords) {
        s.Write(offset);
    }

    if (chunk_size == 0) {
        s.Write<uint64_t>(4);
    } else {
        s.Write<uint64_t>(0);
    }
}

hdf5::expected<BTreeChunkedRawDataNodeKey> BTreeChunkedRawDataNodeKey::DeserializeWithTermInfo(Deserializer& de, ChunkedKeyTerminatorInfo term_info) {
    BTreeChunkedRawDataNodeKey key{};

    key.chunk_size = de.Read<uint32_t>();
    key.filter_mask = de.Read<uint32_t>();
    key.elem_byte_size = term_info.elem_byte_size;

    for (uint8_t i = 0; i < term_info.dimensionality; ++i) {
        key.chunk_offset_in_dataset.coords.push_back(de.Read<uint64_t>());
    }

    auto terminator = de.Read<uint64_t>();

    bool is_unused_key = key.chunk_size == 0;

    // TODO: is terminator always the size of the type?
    if ((is_unused_key && terminator != key.elem_byte_size) || (!is_unused_key && terminator != 0)) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidTerminator, "BTreeChunkedRawDataNodeKey: incorrect terminator");
    }

    return key;
}

template <typename K>
uint16_t BTreeEntries<K>::EntriesUsed() const {
    uint16_t entries_ct = child_pointers.size();

    ASSERT(keys.size() == entries_ct + 1, "Shape of entries was invalid");

    return entries_ct;
}

template <typename K>
uint16_t BTreeEntries<K>::KeySize() const {
    if constexpr (std::is_same_v<K, BTreeGroupNodeKey>) {
        return BTreeGroupNodeKey::kAllocationSize;
    } else if constexpr (std::is_same_v<K, BTreeChunkedRawDataNodeKey>) {
        ASSERT(!keys.empty(), "Cannot determine key size for empty entries");

        return keys.front().AllocationSize();
    } else {
        UNREACHABLE("unsupported key type");

        return {};
    }
}

uint16_t BTreeNode::EntriesUsed() const {
    return cstd::visit([](const auto& entries) {
        return entries.EntriesUsed();
    }, entries);
}

hdf5::expected<cstd::optional<offset_t>> BTreeNode::Get(hdf5::string_view name, FileLink& file, const LocalHeap& heap) const {
    // this does copy the whole node, but it's necessary since future
    // iterations need a place to preserve the lifetime
    BTreeNode current_node = *this;

    for (;;) {
        auto child_index_result = current_node.FindGroupIndex(name, heap, file.io);
        if (!child_index_result) {
            return cstd::unexpected(child_index_result.error());
        }
        cstd::optional<uint16_t> child_index = *child_index_result;

        if (!child_index) {
            return cstd::nullopt;
        }

        const auto& group_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(current_node.entries);

        // leaf node, search for the exact entry
        // pointers point to symbol table entries
        if (current_node.IsLeaf()) {
            return group_entries.child_pointers.at(*child_index);
        }

        // iteratively search the tree - load the next node
        offset_t child_addr = group_entries.child_pointers.at(*child_index);

        file.io.SetPosition(file.superblock.base_addr + child_addr);
        auto child_result = current_node.ReadChild(file.io);
        if (!child_result) return cstd::unexpected(child_result.error());

        // Move to next node for next iteration
        current_node = *child_result;
    }
}

cstd::optional<offset_t> BTreeNode::GetChunk(const ChunkCoordinates& chunk_coords, FileLink& file) const {
    BTreeNode current_node = *this;

    for (;;) {
        cstd::optional<uint16_t> child_index = current_node.FindChunkedIndex(chunk_coords);

        if (!child_index) {
            return cstd::nullopt;
        }

        const auto& chunk_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(current_node.entries);

        // pointers point to raw chunk data
        if (current_node.IsLeaf()) {
            // coordinates must match exactly
            const auto& key = chunk_entries.keys.at(*child_index);
            if (key.chunk_offset_in_dataset.coords == chunk_coords.coords) {
                return chunk_entries.child_pointers.at(*child_index);
            }
            return cstd::nullopt;
        }

        // find the next node
        offset_t child_addr = chunk_entries.child_pointers.at(*child_index);

        file.io.SetPosition(file.superblock.base_addr + child_addr);
        auto child_result = current_node.ReadChild(file.io);
        if (!child_result) return cstd::nullopt;

        current_node = *child_result;
    }
}

template<typename K>
void WriteEntries(const BTreeEntries<K>& entries, Serializer& s) {
    uint16_t entries_ct = entries.child_pointers.size();

    ASSERT(entries.keys.size() == entries_ct + 1, "Shape of entries was invalid");

    for (uint16_t i = 0; i < entries_ct; ++i) {
        s.Write(entries.keys.at(i));
        s.Write(entries.child_pointers.at(i));
    }

    s.Write(entries.keys.back());
}

hdf5::expected<cstd::optional<uint16_t>> BTreeNode::FindGroupIndex(hdf5::string_view key, const LocalHeap& heap, Deserializer& de) const {
    if (!cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        return cstd::nullopt;
    }

    // TODO: can we binary search here?

    const auto& group_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        // empty node, no entries
        return cstd::nullopt;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct + 1;

    auto prev_result = heap.ReadString(group_entries.keys.front().first_object_name, de);
    if (!prev_result) {
        return cstd::unexpected(prev_result.error());
    }

    hdf5::string prev = std::move(*prev_result);

    for (size_t i = 1; i < group_entries.keys.size(); ++i) {
        auto next = heap.ReadString(group_entries.keys[i].first_object_name, de);

        if (!next) {
            return cstd::unexpected(next.error());
        }

        if (prev < key && key <= *next) {
            child_index = i - 1;
            break;
        }

        prev = std::move(*next);
    }

    if (child_index == entries_ct + 1) {
        // name is greater than all keys
        return cstd::nullopt;
    }

    return child_index;
}

hdf5::expected<uint16_t> BTreeNode::GroupInsertionPosition(hdf5::string_view key, const LocalHeap& heap, Deserializer& de) const {
    if (!cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        return hdf5::error(hdf5::HDF5ErrorCode::WrongNodeType, "InsertionPosition only supported for group nodes");
    }

    // TODO: can we binary search here?

    const auto& group_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        return 0;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct;

    for (size_t i = 0; i < entries_ct; ++i) {
        // TODO(cuda_vector): this doesn't need to be allocated
        auto next = heap.ReadString(group_entries.keys.at(i + 1).first_object_name, de);

        if (!next) {
            return cstd::unexpected(next.error());
        }

        if (key <= *next) {
            child_index = i;
            break;
        }
    }

    return child_index;
}

cstd::optional<uint16_t> BTreeNode::FindChunkedIndex(const ChunkCoordinates& chunk_coords) const {
    if (!cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        return cstd::nullopt;
    }

    const auto& chunk_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        // empty node, no entries
        return cstd::nullopt;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct + 1;

    ChunkCoordinates prev = chunk_entries.keys.front().chunk_offset_in_dataset;

    for (size_t i = 1; i < chunk_entries.keys.size(); ++i) {
        ChunkCoordinates next = chunk_entries.keys[i].chunk_offset_in_dataset;
        
        bool is_sentinel = (
            i == entries_ct
            && chunk_entries.keys[i].chunk_size == 0
            && prev == chunk_coords
            && chunk_coords == next
        );
        
        bool in_range = (prev <= chunk_coords && chunk_coords < next);

        if (in_range || is_sentinel) {
            child_index = i - 1;
            break;
        }

        prev = std::move(next);
    }

    if (child_index == entries_ct + 1) {
        // coords are greater than all keys
        return cstd::nullopt;
    }

    return child_index;
}

uint16_t BTreeNode::ChunkedInsertionPosition(const ChunkCoordinates& chunk_coords) const {
    ASSERT(cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries), "ChunkedInsertionPosition only supported for chunked nodes");

    const auto& chunk_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);

    uint16_t entries_ct = EntriesUsed();

    if (entries_ct == 0) {
        return 0;
    }

    // find correct child pointer
    uint16_t child_index = entries_ct;

    for (size_t i = 0; i < entries_ct; ++i) {
        const auto& next = chunk_entries.keys.at(i).chunk_offset_in_dataset;

        if (chunk_coords < next) {
            child_index = i;
            break;
        }
    }

    return child_index;
}

template<typename K>
hdf5::expected<K> BTreeNode::GetMaxKey(FileLink& file) const {
    static_assert(
        std::is_same_v<K, BTreeGroupNodeKey> || std::is_same_v<K, BTreeChunkedRawDataNodeKey>,
        "Unsupported key type"
    );

    ASSERT(cstd::holds_alternative<BTreeEntries<K>>(entries), "GetMaxKey: incorrect key type for this node");

    BTreeNode current_node = *this;

    for (;;) {
        auto node_entries = cstd::get<BTreeEntries<K>>(current_node.entries);

        ASSERT(node_entries.EntriesUsed() != 0, "GetMaxKey called on empty node");

        if (current_node.IsLeaf()) {
            return node_entries.keys.back();
        }

        offset_t rightmost_child = node_entries.child_pointers.back();

        file.io.SetPosition(file.superblock.base_addr + rightmost_child);
        auto child_result = current_node.ReadChild(file.io);
        if (!child_result) return cstd::unexpected(child_result.error());

        current_node = *child_result;
    }
}

template<typename K>
K BTreeNode::GetMinKey() const {
    static_assert(
        std::is_same_v<K, BTreeGroupNodeKey> || std::is_same_v<K, BTreeChunkedRawDataNodeKey>,
        "Unsupported key type"
    );

    ASSERT(cstd::holds_alternative<BTreeEntries<K>>(entries), "GetMinKey: incorrect key type for this node");

    auto node_entries = cstd::get<BTreeEntries<K>>(entries);

    ASSERT(node_entries.EntriesUsed() != 0, "GetMinKey called on empty node");

    return node_entries.keys.front();
}

len_t BTreeNode::AllocationSize(KValues k_val) const {
    const uint16_t k = k_val.Get(IsLeaf());

    uint16_t key_size = cstd::visit([](const auto& entries) -> uint16_t { return entries.KeySize(); }, entries);

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
    if (cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        type = kGroupNodeTy;
    } else if (cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        type = kRawDataChunkNodeTy;
    } else {
        UNREACHABLE("Variant has invalid state");
    }

    s.Write(kSignature);

    s.Write(type);
    s.Write(level);
    s.Write(EntriesUsed());

    s.Write(left_sibling_addr);
    s.Write(right_sibling_addr);

    if (type == kGroupNodeTy) {
        const auto& entr = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(entries);
        WriteEntries(entr, s);
    } else {
        const auto& entr = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);
        WriteEntries(entr, s);
    }
}

hdf5::expected<BTreeNode> BTreeNode::DeserializeGroup(Deserializer& de) {
    if (de.Read<cstd::array<uint8_t, 4>>() != kSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "BTree signature was invalid");
    }

    auto type = de.Read<uint8_t>();

    if (type != kGroupNodeTy && type != kRawDataChunkNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "Invalid BTree node type");
    }

    if (type != kGroupNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "BTreeNode::DeserializeGroup called on non-group node");
    }

    BTreeNode node{};

    node.level = de.Read<uint8_t>();
    auto entries_used = de.Read<uint16_t>();

    node.left_sibling_addr = de.Read<offset_t>();
    node.right_sibling_addr = de.Read<offset_t>();

    BTreeEntries<BTreeGroupNodeKey> entries{};

    for (uint16_t i = 0; i < entries_used; ++i) {
        entries.keys.push_back(de.ReadComplex<BTreeGroupNodeKey>());
        entries.child_pointers.push_back(de.Read<offset_t>());
    }

    entries.keys.push_back(de.ReadComplex<BTreeGroupNodeKey>());

    node.entries = entries;

    return node;
}

hdf5::expected<BTreeNode> BTreeNode::DeserializeChunked(Deserializer& de, ChunkedKeyTerminatorInfo term_info) {
    if (de.Read<cstd::array<uint8_t, 4>>() != kSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "BTree signature was invalid");
    }

    auto type = de.Read<uint8_t>();

    if (type != kGroupNodeTy && type != kRawDataChunkNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "Invalid BTree node type");
    }

    if (type != kRawDataChunkNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "BTreeNode::DeserializeChunked called on non-chunked node");
    }

    BTreeNode node{};

    node.level = de.Read<uint8_t>();
    auto entries_used = de.Read<uint16_t>();

    node.left_sibling_addr = de.Read<offset_t>();
    node.right_sibling_addr = de.Read<offset_t>();

    node.chunked_key_term_info_ = term_info;

    BTreeEntries<BTreeChunkedRawDataNodeKey> entries{};

    for (uint16_t i = 0; i < entries_used; ++i) {
        auto key_result = BTreeChunkedRawDataNodeKey::DeserializeWithTermInfo(de, term_info);
        if (!key_result) return cstd::unexpected(key_result.error());
        entries.keys.push_back(*key_result);
        entries.child_pointers.push_back(de.Read<offset_t>());
    }

    auto last_key_result = BTreeChunkedRawDataNodeKey::DeserializeWithTermInfo(de, term_info);
    if (!last_key_result) return cstd::unexpected(last_key_result.error());
    entries.keys.push_back(*last_key_result);

    node.entries = entries;

    return node;
}

bool BTreeNode::AtCapacity(KValues k) const {
    return EntriesUsed() == k.Get(IsLeaf()) * 2;
}

hdf5::expected<BTreeNode> BTreeNode::ReadChild(Deserializer& de) const {
    if (cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        return DeserializeGroup(de);
    } else if (cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        ASSERT(chunked_key_term_info_.has_value(), "BTreeNode::ReadChild: dimensionality not set for chunked node");

        return DeserializeChunked(de, *chunked_key_term_info_);
    } else {
        UNREACHABLE("Variant has invalid state");
    }
}

BTreeNode BTreeNode::Split(KValues k) {
    return cstd::visit([this, k]<typename Entries>(Entries& l_entries) -> BTreeNode {
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

    ASSERT(EntriesUsed() <= max_entries, "AllocateWriteNewNode called on over-capacity node");

    len_t unused_entries = 2 * k.Get(IsLeaf()) - EntriesUsed();

    // Calculate key size based on node type
    uint16_t key_size = cstd::visit([](const auto& entries) -> uint16_t { return entries.KeySize(); }, entries);

    len_t key_ptr_size = key_size + sizeof(offset_t);

    // intended allocation size
    return written_bytes + unused_entries * key_ptr_size;
};

offset_t BTreeNode::AllocateAndWrite(FileLink& file, KValues k) const {
    len_t alloc_size = AllocationSize(k);
    offset_t alloc_start = file.AllocateAtEOF(alloc_size);

    len_t intended_alloc_size = WriteNodeGetAllocSize(alloc_start, file, k);

    ASSERT(alloc_size == intended_alloc_size, "AllocateWriteNewNode: size mismatch");

    return alloc_start;
}

hdf5::expected<cstd::optional<SplitResult>> BTreeNode::InsertGroup(offset_t this_offset, offset_t name_offset, offset_t obj_header_ptr, FileLink& file, LocalHeap& heap) {
    auto name_str_result = heap.ReadString(name_offset, file.io);
    if (!name_str_result) {
        return cstd::unexpected(name_str_result.error());
    }
    hdf5::string_view name_str = *name_str_result;

    const KValues k {
        .leaf = file.superblock.group_leaf_node_k,
        .internal = file.superblock.group_internal_node_k,
    };

    auto RawInsert = [&file, &heap](BTreeNode& node, BTreeGroupNodeKey key, offset_t child_ptr) -> hdf5::expected<void> {
        auto key_str = heap.ReadString(key.first_object_name, file.io);

        if (!key_str) {
            return cstd::unexpected(key_str.error());
        }

        auto& ins_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(node.entries);
        auto ins_pos_result = node.GroupInsertionPosition(*key_str, heap, file.io);
        if (!ins_pos_result) {
            return cstd::unexpected(ins_pos_result.error());
        }
        uint16_t ins_pos = *ins_pos_result;

        ins_entries.child_pointers.insert(
            ins_entries.child_pointers.begin() + ins_pos,
            child_ptr
        );

        ins_entries.keys.insert( // keys are offset by one
            ins_entries.keys.begin() + ins_pos + 1,
            key
        );

        return {};
    };

    struct StackFrame {
        BTreeNode node;
        offset_t node_offset;
    };

    cstd::inplace_vector<StackFrame, kMaxDepth> path_stack;

    // 1: descend to the correct leaf node
    BTreeNode current_node = *this;
    offset_t current_offset = this_offset;

    while (!current_node.IsLeaf()) {
        path_stack.push_back({current_node, current_offset});

        auto& g_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(current_node.entries);
        auto child_idx_result = current_node.FindGroupIndex(name_str, heap, file.io);
        if (!child_idx_result) {
            return cstd::unexpected(child_idx_result.error());
        }
        cstd::optional<uint16_t> child_idx = *child_idx_result;

        if (!child_idx) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "BTreeNode::InsertGroup: could not find child index");
        }

        offset_t child_offset = g_entries.child_pointers.at(*child_idx);

        file.io.SetPosition(child_offset);
        auto child_result = current_node.ReadChild(file.io);
        if (!child_result) {
            return cstd::unexpected(child_result.error());
        }

        current_node = *child_result;
        current_offset = child_offset;
    }

    // 2: insert into the leaf node
    auto& g_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(current_node.entries);
    cstd::optional<SplitResult> split_to_propagate{};

    if (current_node.AtCapacity(k)) {
        uint16_t mid_index = k.leaf;
        BTreeGroupNodeKey promoted_key = g_entries.keys.at(mid_index);
        BTreeNode new_node = current_node.Split(k);

        auto promoted_key_str = heap.ReadString(promoted_key.first_object_name, file.io);
        if (!promoted_key_str) {
            return cstd::unexpected(promoted_key_str.error());
        }

        if (name_str <= *promoted_key_str) {
            auto insert_result = RawInsert(current_node, { name_offset }, obj_header_ptr);
            if (!insert_result) {
                return cstd::unexpected(insert_result.error());
            }
        } else {
            auto insert_result = RawInsert(new_node, { name_offset }, obj_header_ptr);
            if (!insert_result) {
                return cstd::unexpected(insert_result.error());
            }
        }

        offset_t new_node_alloc = new_node.AllocateAndWrite(file, k);

        split_to_propagate = {
            .promoted_key = promoted_key,
            .new_node_offset = new_node_alloc,
        };
    } else {
        auto insert_result = RawInsert(current_node, { name_offset }, obj_header_ptr);
        if (!insert_result) {
            return cstd::unexpected(insert_result.error());
        }
    }

    current_node.WriteNodeGetAllocSize(current_offset, file, k);

    // 3: propagate splits back up the path
    while (!path_stack.empty() && split_to_propagate.has_value()) {
        auto& frame = path_stack.back();
        BTreeNode parent_node = frame.node;
        offset_t parent_offset = frame.node_offset;
        path_stack.pop_back();

        auto& parent_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(parent_node.entries);

        if (parent_node.AtCapacity(k)) {
            uint16_t mid_index = k.internal;
            BTreeGroupNodeKey promoted_key = parent_entries.keys.at(mid_index);
            BTreeNode new_node = parent_node.Split(k);

            auto promoted_key_str = heap.ReadString(promoted_key.first_object_name, file.io);
            if (!promoted_key_str) {
                return cstd::unexpected(promoted_key_str.error());
            }

            if (name_str <= *promoted_key_str) {
                auto insert_result = RawInsert(parent_node, split_to_propagate->promoted_key, split_to_propagate->new_node_offset);
                if (!insert_result) {
                    return cstd::unexpected(insert_result.error());
                }
            } else {
                auto insert_result = RawInsert(new_node, split_to_propagate->promoted_key, split_to_propagate->new_node_offset);
                if (!insert_result) {
                    return cstd::unexpected(insert_result.error());
                }
            }

            offset_t new_node_alloc = new_node.AllocateAndWrite(file, k);

            split_to_propagate = {
                .promoted_key = promoted_key,
                .new_node_offset = new_node_alloc,
            };
        } else {
            auto insert_result = RawInsert(parent_node, split_to_propagate->promoted_key, split_to_propagate->new_node_offset);
            if (!insert_result) {
                return cstd::unexpected(insert_result.error());
            }
            split_to_propagate = cstd::nullopt;
        }

        parent_node.WriteNodeGetAllocSize(parent_offset, file, k);
    }

    return split_to_propagate;
}

// TODO(expected): maybe just return expected?
// TODO: refactor this and InsertGroup to use the same internals; currently lot of duplicated logic
hdf5::expected<cstd::optional<SplitResultChunked>> BTreeNode::InsertChunked(
    offset_t this_offset,
    const BTreeChunkedRawDataNodeKey& key,
    offset_t data_ptr,
    FileLink& file
) {
    const KValues k {
        .leaf = kChunkedRawDataK,
        .internal = kChunkedRawDataK,
    };

    auto RawInsert = [](BTreeNode& node, const BTreeChunkedRawDataNodeKey& insert_key, offset_t child_ptr) -> hdf5::expected<void> {
        auto& ins_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(node.entries);
        uint16_t ins_pos = node.ChunkedInsertionPosition(insert_key.chunk_offset_in_dataset);

        ins_entries.child_pointers.insert(
            ins_entries.child_pointers.begin() + ins_pos,
            child_ptr
        );

        ins_entries.keys.insert( // keys aren't offset by one for chunked
            ins_entries.keys.begin() + ins_pos,
            insert_key
        );

        return {};
    };

    struct StackFrame {
        BTreeNode node;
        offset_t node_offset;
    };

    cstd::inplace_vector<StackFrame, kMaxDepth> path_stack;

    // 1: descend to the correct leaf node
    BTreeNode current_node = *this;
    offset_t current_offset = this_offset;

    while (!current_node.IsLeaf()) {
        path_stack.push_back({current_node, current_offset});

        cstd::optional<uint16_t> child_idx = current_node.FindChunkedIndex(key.chunk_offset_in_dataset);

        if (!child_idx) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidDataValue, "BTreeNode::InsertChunked: could not find child index");
        }

        auto& c_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(current_node.entries);
        offset_t child_offset = c_entries.child_pointers.at(*child_idx);

        file.io.SetPosition(file.superblock.base_addr + child_offset);
        auto child_result = current_node.ReadChild(file.io);
        if (!child_result) {
            return cstd::unexpected(child_result.error());
        }

        current_node = *child_result;
        current_offset = child_offset;
    }

    // Phase 2: Insert into the leaf node
    auto& c_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(current_node.entries);
    cstd::optional<SplitResultChunked> split_to_propagate{};

    if (current_node.AtCapacity(k)) {
        uint16_t mid_index = k.leaf;
        BTreeChunkedRawDataNodeKey promoted_key = c_entries.keys.at(mid_index);
        BTreeNode new_node = current_node.Split(k);

        if (key.chunk_offset_in_dataset < promoted_key.chunk_offset_in_dataset) {
            RawInsert(current_node, key, data_ptr);
        } else {
            RawInsert(new_node, key, data_ptr);
        }

        offset_t new_node_alloc = new_node.AllocateAndWrite(file, k);

        split_to_propagate = SplitResultChunked {
            .promoted_key = promoted_key,
            .new_node_offset = new_node_alloc,
        };
    } else {
        RawInsert(current_node, key, data_ptr);
    }

    current_node.WriteNodeGetAllocSize(current_offset, file, k);

    // 3: propagate splits back up the path
    while (!path_stack.empty() && split_to_propagate.has_value()) {
        auto& frame = path_stack.back();
        BTreeNode parent_node = frame.node;
        offset_t parent_offset = frame.node_offset;
        path_stack.pop_back();

        auto& parent_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(parent_node.entries);

        if (parent_node.AtCapacity(k)) {
            uint16_t mid_index = k.internal;
            BTreeChunkedRawDataNodeKey promoted_key = parent_entries.keys.at(mid_index);
            BTreeNode new_node = parent_node.Split(k);

            if (key.chunk_offset_in_dataset < promoted_key.chunk_offset_in_dataset) {
                RawInsert(parent_node, split_to_propagate->promoted_key, split_to_propagate->new_node_offset);
            } else {
                RawInsert(new_node, split_to_propagate->promoted_key, split_to_propagate->new_node_offset);
            }

            offset_t new_node_alloc = new_node.AllocateAndWrite(file, k);

            split_to_propagate = SplitResultChunked {
                .promoted_key = promoted_key,
                .new_node_offset = new_node_alloc,
            };
        } else {
            RawInsert(parent_node, split_to_propagate->promoted_key, split_to_propagate->new_node_offset);

            split_to_propagate = cstd::nullopt;
        }

        parent_node.WriteNodeGetAllocSize(parent_offset, file, k);
    }

    return split_to_propagate;
}

hdf5::expected<cstd::optional<offset_t>> GroupBTree::Get(hdf5::string_view name) const {
    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        return cstd::nullopt;
    }

    return (*root_result)->Get(name, *file_, heap_);
}

hdf5::expected<void> GroupBTree::InsertGroup(offset_t name_offset, offset_t object_header_ptr) {
    const BTreeNode::KValues k {
        .leaf = file_->superblock.group_leaf_node_k,
        .internal = file_->superblock.group_internal_node_k
    };

    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        BTreeEntries<BTreeGroupNodeKey> entries{};

        auto empty_str_offset_result = heap_.WriteString("", *file_);
        if (!empty_str_offset_result) {
            return cstd::unexpected(empty_str_offset_result.error());
        }
        offset_t empty_str_offset = *empty_str_offset_result;

        entries.keys.push_back({ empty_str_offset });
        entries.child_pointers.push_back(/* root: */ object_header_ptr);
        entries.keys.push_back({ name_offset });

        BTreeNode new_root {
            .level = 0,
            .entries = entries,
        };

        addr_ = new_root.AllocateAndWrite(*file_, k);

        return {};
    }

    auto split_result = (*root_result)->InsertGroup(*addr_, name_offset, object_header_ptr, *file_, heap_);
    if (!split_result) {
        return cstd::unexpected(split_result.error());
    }
    cstd::optional<SplitResult> split = *split_result;

    if (split.has_value()) {
        BTreeEntries<BTreeGroupNodeKey> entries{};

        auto min = (*root_result)->GetMinKey<BTreeGroupNodeKey>();
        auto max_result = (*root_result)->GetMaxKey<BTreeGroupNodeKey>(*file_);
        if (!max_result) {
            return cstd::unexpected(max_result.error());
        }

        entries.keys.push_back(min);
        entries.child_pointers.push_back(/* root: */ *addr_);
        entries.keys.push_back(split->promoted_key);
        entries.child_pointers.push_back(split->new_node_offset);
        entries.keys.push_back(*max_result);

        if ((*root_result)->level == std::numeric_limits<uint8_t>::max()) {
            return hdf5::error(hdf5::HDF5ErrorCode::BTreeOverflow, "BTree level overflow");
        }

        BTreeNode new_root {
            .level = static_cast<uint8_t>((*root_result)->level + 1),
            .entries = entries,
        };

        addr_ = new_root.AllocateAndWrite(*file_, k);
    }

    return {};
}

hdf5::expected<void> ChunkedBTree::InsertChunk(const ChunkCoordinates& chunk_coords, uint32_t chunk_size, uint32_t filter_mask, offset_t data_ptr) {
    const BTreeNode::KValues k {
        .leaf = BTreeNode::kChunkedRawDataK,
        .internal = BTreeNode::kChunkedRawDataK
    };

    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    BTreeChunkedRawDataNodeKey new_key {
        .chunk_size = chunk_size,
        .filter_mask = filter_mask,
        .chunk_offset_in_dataset = chunk_coords
    };

    ASSERT(root_result->has_value(), "should have been created elsewhere");

    auto split_result = (*root_result)->InsertChunked(*addr_, new_key, data_ptr, *file_);
    if (!split_result) return cstd::unexpected(split_result.error());
    cstd::optional<SplitResultChunked> split = *split_result;

    if (split.has_value()) {
        BTreeEntries<BTreeChunkedRawDataNodeKey> entries{};

        auto min = (*root_result)->GetMinKey<BTreeChunkedRawDataNodeKey>();
        auto max_result = (*root_result)->GetMaxKey<BTreeChunkedRawDataNodeKey>(*file_);
        if (!max_result) {
            return cstd::unexpected(max_result.error());
        }

        entries.keys.push_back(min);
        entries.child_pointers.push_back(*addr_);
        entries.keys.push_back(split->promoted_key);
        entries.child_pointers.push_back(split->new_node_offset);
        entries.keys.push_back(*max_result);

        if ((*root_result)->level == std::numeric_limits<uint8_t>::max()) {
            return hdf5::error(hdf5::HDF5ErrorCode::BTreeOverflow, "BTree level overflow");
        }

        BTreeNode new_root {
            .level = static_cast<uint8_t>((*root_result)->level + 1),
            .entries = entries,
        };

        addr_ = new_root.AllocateAndWrite(*file_, k);
    }

    return {};
}

hdf5::expected<cstd::optional<offset_t>> ChunkedBTree::GetChunk(const ChunkCoordinates& chunk_coords) const {
    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        return cstd::nullopt;
    }

    return (*root_result)->GetChunk(chunk_coords, *file_);
}

hdf5::expected<std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> ChunkedBTree::Offsets() const {
    std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>> result{};

    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        return result;
    }

    auto recurse_result = (*root_result)->RecurseChunked([&result](const BTreeChunkedRawDataNodeKey& key, offset_t data_offset) -> hdf5::expected<void> {
        result.emplace_back(key.chunk_offset_in_dataset, data_offset, key.chunk_size);
        return {};
    }, *file_);
    if (!recurse_result) return cstd::unexpected(recurse_result.error());

    return result;
}

offset_t ChunkedBTree::CreateNew(const std::shared_ptr<FileLink>& file, const hdf5::dim_vector<uint64_t>& max_size) {
    BTreeNode::KValues k{
        .leaf = BTreeNode::kChunkedRawDataK,
        .internal = BTreeNode::kChunkedRawDataK
    };

    BTreeEntries<BTreeChunkedRawDataNodeKey> entries{};

    entries.keys.push_back({
        .chunk_size = 0,
        .filter_mask = 0,
        .chunk_offset_in_dataset = ChunkCoordinates(max_size),
    });

    return BTreeNode { .level = 0, .entries = entries }.AllocateAndWrite(*file, k);
}

hdf5::expected<cstd::optional<BTreeNode>> ChunkedBTree::ReadRoot() const {
    if (!addr_.has_value()) {
        return cstd::nullopt;
    }

    file_->io.SetPosition(file_->superblock.base_addr + *addr_);

    return BTreeNode::DeserializeChunked(file_->io, terminator_info_);
}

hdf5::expected<size_t> GroupBTree::Size() const {
    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        return 0;
    }

    size_t size = 0;

    auto recurse_result = (*root_result)->Recurse([&size](const hdf5::string&, offset_t) -> hdf5::expected<void> {
        ++size;
        return {};
    }, *file_);
    if (!recurse_result) return cstd::unexpected(recurse_result.error());

    return size;
}

hdf5::expected<cstd::inplace_vector<offset_t, GroupBTree::kMaxGroupElements>> GroupBTree::Elements() const {
    auto root_result = ReadRoot();
    if (!root_result) return cstd::unexpected(root_result.error());

    if (!root_result->has_value()) {
        return cstd::inplace_vector<offset_t, kMaxGroupElements>{};
    }

    cstd::inplace_vector<offset_t, kMaxGroupElements> elems;

    auto recurse_result = (*root_result)->Recurse([&elems](const hdf5::string&, offset_t ptr) -> hdf5::expected<void> {
        if (elems.size() >= elems.capacity()) {
            return hdf5::error(hdf5::HDF5ErrorCode::CapacityExceeded, "Group has too many elements (exceeds kMaxGroupElements)");
        }
        elems.push_back(ptr);
        return {};
    }, *file_);
    if (!recurse_result) return cstd::unexpected(recurse_result.error());

    return elems;
}

hdf5::expected<cstd::optional<BTreeNode>> GroupBTree::ReadRoot() const {
    if (!addr_.has_value()) {
        return cstd::nullopt;
    }

    file_->io.SetPosition(*addr_);

    return BTreeNode::DeserializeGroup(file_->io);
}
