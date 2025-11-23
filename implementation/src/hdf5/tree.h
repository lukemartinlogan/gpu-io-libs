#pragma once
#include <array>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "file_link.h"
#include "local_heap.h"
#include "object.h"
#include "types.h"
#include "../serialization/serialization.h"
#include "gpu_string.h"
#include "../util/gpu_vector.h"

struct BTreeGroupNodeKey {
    // byte offset into local heap
    // first object name in the subtree the key describes
    len_t first_object_name;

    static constexpr len_t kAllocationSize = sizeof(len_t);
};

static_assert(serde::TriviallySerializable<BTreeGroupNodeKey>);

static_assert(
    BTreeGroupNodeKey::kAllocationSize == sizeof(BTreeGroupNodeKey),
    "no extra fields should be added to key"
);

struct ChunkCoordinates {
    hdf5::dim_vector<uint64_t> coords;

    ChunkCoordinates() = default;

    explicit ChunkCoordinates(const hdf5::dim_vector<uint64_t>& coordinates) : coords(coordinates) {}
    explicit ChunkCoordinates(hdf5::dim_vector<uint64_t>&& coordinates) : coords(std::move(coordinates)) {}

   bool operator==(const ChunkCoordinates& other) const {
       return coords == other.coords;
   }

   bool operator!=(const ChunkCoordinates& other) const {
       return !(*this == other);
   }

   bool operator<(const ChunkCoordinates& other) const {
       return coords < other.coords;
   }

   bool operator<=(const ChunkCoordinates& other) const {
       return other >= *this;
   }

   bool operator>(const ChunkCoordinates& other) const {
       return other < *this;
   }

   bool operator>=(const ChunkCoordinates& other) const {
       return !(*this < other);
   }

    [[nodiscard]] size_t Dimensions() const {
        return coords.size();
    }

    template<serde::Serializer S>
    void Serialize(S& s) const {
        serde::Write(s, static_cast<uint64_t>(coords.size()));
        for (const uint64_t coord: coords) {
            serde::Write(s, coord);
        }
    }

    template<serde::Deserializer D>
    static ChunkCoordinates Deserialize(D& de) {
        ChunkCoordinates chunk_coords{};

        uint64_t dim_ct = serde::Read<D, uint64_t>(de);

        for (uint64_t i = 0; i < dim_ct; ++i) {
            chunk_coords.coords.push_back(serde::Read<D, uint64_t>(de));
        }
        return chunk_coords;
    }
};

struct ChunkedKeyTerminatorInfo {
    uint8_t dimensionality;
    uint64_t elem_byte_size;
};

struct BTreeChunkedRawDataNodeKey {
    // in bytes
    uint32_t chunk_size;
    // bit field indicating which filters have been skipped for this chunk
    uint32_t filter_mask;
    // .size() == number of dimensions
    // extra uint64_t(0) at the end (not stored)
    ChunkCoordinates chunk_offset_in_dataset;

    size_t elem_byte_size;

    template<serde::Serializer S>
    void Serialize(S& s) const {
        serde::Write(s, chunk_size);
        serde::Write(s, filter_mask);

        for (const uint64_t offset: chunk_offset_in_dataset.coords) {
            serde::Write(s, offset);
        }

        if (chunk_size == 0) {
            serde::Write(s, static_cast<uint64_t>(4));
        } else {
            serde::Write(s, static_cast<uint64_t>(0));
        }
    }

    template<serde::Deserializer D>
    static hdf5::expected<BTreeChunkedRawDataNodeKey> DeserializeWithTermInfo(D& de, ChunkedKeyTerminatorInfo term_info) {
        BTreeChunkedRawDataNodeKey key{};

        key.chunk_size = serde::Read<D, uint32_t>(de);
        key.filter_mask = serde::Read<D, uint32_t>(de);
        key.elem_byte_size = term_info.elem_byte_size;

        for (uint8_t i = 0; i < term_info.dimensionality; ++i) {
            key.chunk_offset_in_dataset.coords.push_back(serde::Read<D, uint64_t>(de));
        }

        auto terminator = serde::Read<D, uint64_t>(de);

        bool is_unused_key = key.chunk_size == 0;

        // TODO: is terminator always the size of the type?
        if ((is_unused_key && terminator != key.elem_byte_size) || (!is_unused_key && terminator != 0)) {
            return hdf5::error(hdf5::HDF5ErrorCode::InvalidTerminator, "BTreeChunkedRawDataNodeKey: incorrect terminator");
        }

        return key;
    }

    [[nodiscard]] uint16_t AllocationSize() const {
        // Key size = chunk_size + filter_mask + (dimensions * sizeof(uint64_t))
        // + 1 for the terminating 0, since this is allocation size
        uint16_t dimensions = static_cast<uint16_t>(chunk_offset_in_dataset.Dimensions());
        return sizeof(uint32_t) + sizeof(uint32_t) + (dimensions + 1) * sizeof(uint64_t);
    }
};

template<typename K>
struct BTreeEntries {
private:
    static constexpr size_t MAX_BTREE_ENTRIES = 32;

public:
    // TODO: enforce child_pointers.size() + 1 == keys.size()
    cstd::inplace_vector<K, MAX_BTREE_ENTRIES> keys;
    cstd::inplace_vector<offset_t, MAX_BTREE_ENTRIES> child_pointers;

    [[nodiscard]] uint16_t EntriesUsed() const;

    [[nodiscard]] uint16_t KeySize() const;

private:
    static_assert(
        std::is_same_v<K, BTreeGroupNodeKey> || std::is_same_v<K, BTreeChunkedRawDataNodeKey>,
        "Unsupported key type"
    );
};

struct SplitResult;
struct SplitResultChunked;

struct BTreeNode {
    // type: (not stored, check variant)
    // implies max degree K of the tree & size of each key field

    // what level node appears in the tree, leaf nodes are at zero
    // indicates if child pointers point to subtrees or to data
    uint8_t level{};
    // relative addr of curr node's left sibling
    // if leftmost, then kUndefinedOffset
    offset_t left_sibling_addr = kUndefinedOffset;
    // relative addr of curr node's right sibling
    // if rightmost, then kUndefinedOffset
    offset_t right_sibling_addr = kUndefinedOffset;
    // last entry's child_pointer field is unused
    cstd::variant<
        BTreeEntries<BTreeGroupNodeKey>,
        BTreeEntries<BTreeChunkedRawDataNodeKey>
    > entries{};

    cstd::optional<ChunkedKeyTerminatorInfo> chunked_key_term_info_{};

    // max number of children this node points to
    // all nodes have same max degree (max entries used) but
    // most nodes point to less than that
    [[nodiscard]] uint16_t EntriesUsed() const;

    [[nodiscard]] bool IsLeaf() const {
        return level == 0;
    }

    [[nodiscard]] hdf5::expected<cstd::optional<offset_t>> Get(hdf5::string_view name, FileLink& file, const LocalHeap& heap) const;

    [[nodiscard]] cstd::optional<offset_t> GetChunk(const ChunkCoordinates& chunk_coords, FileLink& file) const;

    template<serde::Serializer S>
    void Serialize(S& s) const;

    template<serde::Deserializer D>
    static hdf5::expected<BTreeNode> DeserializeGroup(D& de);

    template<serde::Deserializer D>
    static hdf5::expected<BTreeNode> DeserializeChunked(D& de, ChunkedKeyTerminatorInfo term_info);

private:
    friend struct GroupBTree;
    friend struct ChunkedBTree;
    friend class Group;

    struct KValues {
        uint16_t leaf;
        uint16_t internal;

        [[nodiscard]] uint16_t Get(bool is_leaf) const {
            return is_leaf ? leaf : internal;
        }
    };

    template<serde::Deserializer D>
    hdf5::expected<BTreeNode> ReadChild(D& de) const {
        if (cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
            return DeserializeGroup(de);
        } else if (cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
            ASSERT(chunked_key_term_info_.has_value(), "BTreeNode::ReadChild: dimensionality not set for chunked node");

            return DeserializeChunked(de, *chunked_key_term_info_);
        } else {
            UNREACHABLE("Variant has invalid state");
            return {};
        }
    }

    [[nodiscard]] BTreeNode Split(KValues k);

    hdf5::expected<cstd::optional<SplitResult>> InsertGroup(offset_t this_offset, offset_t name_offset, offset_t obj_header_ptr, FileLink& file, LocalHeap& heap);

    hdf5::expected<cstd::optional<SplitResultChunked>> InsertChunked(
        offset_t this_offset,
        const BTreeChunkedRawDataNodeKey& key,
        offset_t data_ptr,
        FileLink& file
    );

    template<serde::Deserializer D>
    hdf5::expected<cstd::optional<uint16_t>> FindGroupIndex(hdf5::string_view key, const LocalHeap& heap, D& de) const {
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

    [[nodiscard]] cstd::optional<uint16_t> FindChunkedIndex(const ChunkCoordinates& chunk_coords) const;

    [[nodiscard]] bool AtCapacity(KValues k) const;

    template<serde::Deserializer D>
    hdf5::expected<uint16_t> GroupInsertionPosition(hdf5::string_view key, const LocalHeap& heap, D& de) const {
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

    [[nodiscard]] uint16_t ChunkedInsertionPosition(const ChunkCoordinates& chunk_coords) const;

    template <class K>
    [[nodiscard]] hdf5::expected<K> GetMaxKey(FileLink& file) const;

    template <class K>
    [[nodiscard]] K GetMinKey() const;

    [[nodiscard]] len_t AllocationSize(KValues k) const;

    len_t WriteNodeGetAllocSize(offset_t offset, FileLink& file, KValues k) const;

    offset_t AllocateAndWrite(FileLink& file, KValues k) const;

    template<typename Visitor>
    hdf5::expected<void> Recurse(Visitor&& visitor, FileLink& file) const {
        static_assert(
            std::is_invocable_r_v<hdf5::expected<void>, Visitor, hdf5::string, offset_t>,
            "Visitor must be invocable with (hdf5::string, offset_t) and return hdf5::expected<void>"
        );

        ASSERT(cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries), "Recurse only supported for group nodes");

        // Stack frame to track nodes and their current child index
        struct StackFrame {
            BTreeNode node;
            size_t current_index;
        };

        cstd::inplace_vector<StackFrame, kMaxDepth> stack;
        stack.push_back({*this, 0});

        while (!stack.empty()) {
            auto& frame = stack.back();
            auto g_entries = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(frame.node.entries);

            // If we've processed all children of this node, pop it
            if (frame.current_index >= g_entries.EntriesUsed()) {
                stack.pop_back();
                continue;
            }

            offset_t ptr = g_entries.child_pointers.at(frame.current_index);
            size_t current_idx = frame.current_index;
            ++frame.current_index;

            if (frame.node.IsLeaf()) {
                auto name_result = hdf5::to_string(g_entries.keys.at(current_idx).first_object_name);

                if (!name_result) {
                    return cstd::unexpected(name_result.error());
                }

                auto visitor_result = visitor(std::move(*name_result), ptr);
                if (!visitor_result) {
                    return cstd::unexpected(visitor_result.error());
                }
            } else {
                file.io.SetPosition(file.superblock.base_addr + ptr);
                auto child_result = frame.node.ReadChild(file.io);
                if (!child_result) return cstd::unexpected(child_result.error());

                stack.push_back({*child_result, 0});
            }
        }

        return {};
    }

    template<typename Visitor>
    hdf5::expected<void> RecurseChunked(Visitor&& visitor, FileLink& file) const {
        static_assert(
            std::is_invocable_r_v<hdf5::expected<void>, Visitor, BTreeChunkedRawDataNodeKey, offset_t>,
            "Visitor must be invocable with (BTreeChunkedRawDataNodeKey, offset_t) and return hdf5::expected<void>"
        );

        ASSERT(cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries), "RecurseChunked only supported for chunked nodes");

        struct StackFrame {
            BTreeNode node;
            size_t current_index;
        };

        cstd::inplace_vector<StackFrame, kMaxDepth> stack;
        stack.push_back({*this, 0});

        while (!stack.empty()) {
            auto& frame = stack.back();
            auto c_entries = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(frame.node.entries);

            if (frame.current_index >= c_entries.EntriesUsed()) {
                stack.pop_back();
                continue;
            }

            offset_t ptr = c_entries.child_pointers.at(frame.current_index);
            size_t current_idx = frame.current_index;
            ++frame.current_index;

            if (frame.node.IsLeaf()) {
                const auto& key = c_entries.keys.at(current_idx);

                // Only visit chunks that actually exist (chunk_size > 0)
                if (key.chunk_size > 0) {
                    auto visitor_result = visitor(key, ptr);
                    if (!visitor_result) {
                        return cstd::unexpected(visitor_result.error());
                    }
                }
            } else {
                file.io.SetPosition(file.superblock.base_addr + ptr);
                auto child_result = frame.node.ReadChild(file.io);
                if (!child_result) return cstd::unexpected(child_result.error());

                stack.push_back({*child_result, 0});
            }
        }

        return {};
    }

private:
    static constexpr uint8_t kGroupNodeTy = 0, kRawDataChunkNodeTy = 1;
    static constexpr uint16_t kChunkedRawDataK = 32;
    static constexpr cstd::array<uint8_t, 4> kSignature = { 'T', 'R', 'E', 'E' };

    static constexpr size_t kMaxDepth = 16;
};

struct SplitResult {
    BTreeGroupNodeKey promoted_key;
    offset_t new_node_offset;
};

struct SplitResultChunked {
    BTreeChunkedRawDataNodeKey promoted_key;
    offset_t new_node_offset;
};

struct GroupBTree {
    // TODO: max need to increase this
    static constexpr size_t kMaxGroupElements = 128;

    explicit GroupBTree(offset_t addr, std::shared_ptr<FileLink> file, const LocalHeap& heap)
        : file_(std::move(file)), heap_(heap), addr_(addr) {}

    [[nodiscard]] hdf5::expected<cstd::optional<offset_t>> Get(hdf5::string_view name) const;

    hdf5::expected<void> InsertGroup(offset_t name_offset, offset_t object_header_ptr);

    [[nodiscard]] hdf5::expected<size_t> Size() const;

    [[nodiscard]] hdf5::expected<cstd::inplace_vector<offset_t, kMaxGroupElements>> Elements() const;
private:
    friend class Group;

    GroupBTree() = default;

    [[nodiscard]] hdf5::expected<cstd::optional<BTreeNode>> ReadRoot() const;

private:
    std::shared_ptr<FileLink> file_{};
    LocalHeap heap_{};
    cstd::optional<offset_t> addr_{};
};

struct ChunkedBTree {
    explicit ChunkedBTree(offset_t addr, std::shared_ptr<FileLink> file, ChunkedKeyTerminatorInfo term_info)
        : file_(std::move(file)), addr_(addr), terminator_info_(term_info) {}

    hdf5::expected<void> InsertChunk(
        const ChunkCoordinates& chunk_coords,
        uint32_t chunk_size,
        uint32_t filter_mask,
        offset_t data_ptr
    );

    [[nodiscard]] hdf5::expected<cstd::optional<offset_t>> GetChunk(const ChunkCoordinates& chunk_coords) const;

    // can be very large, so use dynamic vector
    [[nodiscard]] hdf5::expected<hdf5::gpu_vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> Offsets() const;

    static offset_t CreateNew(const std::shared_ptr<FileLink>& file, const hdf5::dim_vector<uint64_t>& max_size);

private:
    ChunkedBTree() = default;

public:
    [[nodiscard]] hdf5::expected<cstd::optional<BTreeNode>> ReadRoot() const;

private:
    std::shared_ptr<FileLink> file_{};
    cstd::optional<offset_t> addr_{};

    ChunkedKeyTerminatorInfo terminator_info_{};
};

// these implementations are here because of declaration order
template<typename K, serde::Serializer S>
void WriteEntries(const BTreeEntries<K>& entries, S& s) {
    uint16_t entries_ct = entries.child_pointers.size();

    ASSERT(entries.keys.size() == entries_ct + 1, "Shape of entries was invalid");

    for (uint16_t i = 0; i < entries_ct; ++i) {
        serde::Write(s, entries.keys.at(i));
        serde::Write(s, entries.child_pointers.at(i));
    }

    serde::Write(s, entries.keys.back());
}

template<serde::Serializer S>
void BTreeNode::Serialize(S& s) const {
    uint8_t type;
    if (cstd::holds_alternative<BTreeEntries<BTreeGroupNodeKey>>(entries)) {
        type = kGroupNodeTy;
    } else if (cstd::holds_alternative<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries)) {
        type = kRawDataChunkNodeTy;
    } else {
        UNREACHABLE("Variant has invalid state");
    }

    serde::Write(s, kSignature);

    serde::Write(s, type);
    serde::Write(s, level);
    serde::Write(s, EntriesUsed());

    serde::Write(s, left_sibling_addr);
    serde::Write(s, right_sibling_addr);

    if (type == kGroupNodeTy) {
        const auto& entr = cstd::get<BTreeEntries<BTreeGroupNodeKey>>(entries);
        WriteEntries(entr, s);
    } else {
        const auto& entr = cstd::get<BTreeEntries<BTreeChunkedRawDataNodeKey>>(entries);
        WriteEntries(entr, s);
    }
}

template<serde::Deserializer D>
hdf5::expected<BTreeNode> BTreeNode::DeserializeGroup(D& de) {
    if (serde::Read<D, cstd::array<uint8_t, 4>>(de) != kSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "BTree signature was invalid");
    }

    auto type = serde::Read<D, uint8_t>(de);

    if (type != kGroupNodeTy && type != kRawDataChunkNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "Invalid BTree node type");
    }

    if (type != kGroupNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "BTreeNode::DeserializeGroup called on non-group node");
    }

    BTreeNode node{};

    node.level = serde::Read<D, uint8_t>(de);
    auto entries_used = serde::Read<D, uint16_t>(de);

    node.left_sibling_addr = serde::Read<D, offset_t>(de);
    node.right_sibling_addr = serde::Read<D, offset_t>(de);

    BTreeEntries<BTreeGroupNodeKey> entries{};

    for (uint16_t i = 0; i < entries_used; ++i) {
        entries.keys.push_back(serde::Read<D, BTreeGroupNodeKey>(de));
        entries.child_pointers.push_back(serde::Read<D, offset_t>(de));
    }

    entries.keys.push_back(serde::Read<D, BTreeGroupNodeKey>(de));

    node.entries = entries;

    return node;
}

template<serde::Deserializer D>
hdf5::expected<BTreeNode> BTreeNode::DeserializeChunked(D& de, ChunkedKeyTerminatorInfo term_info) {
    if (serde::Read<D, cstd::array<uint8_t, 4>>(de) != kSignature) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidSignature, "BTree signature was invalid");
    }

    auto type = serde::Read<D, uint8_t>(de);

    if (type != kGroupNodeTy && type != kRawDataChunkNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "Invalid BTree node type");
    }

    if (type != kRawDataChunkNodeTy) {
        return hdf5::error(hdf5::HDF5ErrorCode::InvalidType, "BTreeNode::DeserializeChunked called on non-chunked node");
    }

    BTreeNode node{};

    node.level = serde::Read<D, uint8_t>(de);
    auto entries_used = serde::Read<D, uint16_t>(de);

    node.left_sibling_addr = serde::Read<D, offset_t>(de);
    node.right_sibling_addr = serde::Read<D, offset_t>(de);

    node.chunked_key_term_info_ = term_info;

    BTreeEntries<BTreeChunkedRawDataNodeKey> entries{};

    for (uint16_t i = 0; i < entries_used; ++i) {
        auto key_result = BTreeChunkedRawDataNodeKey::DeserializeWithTermInfo(de, term_info);
        if (!key_result) return cstd::unexpected(key_result.error());
        entries.keys.push_back(*key_result);
        entries.child_pointers.push_back(serde::Read<D, offset_t>(de));
    }

    auto last_key_result = BTreeChunkedRawDataNodeKey::DeserializeWithTermInfo(de, term_info);
    if (!last_key_result) return cstd::unexpected(last_key_result.error());
    entries.keys.push_back(*last_key_result);

    node.entries = entries;

    return node;
}