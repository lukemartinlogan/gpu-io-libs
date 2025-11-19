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

    static constexpr len_t kAllocationSize = sizeof(len_t);
};

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

    void Serialize(Serializer& s) const;

    static hdf5::expected<BTreeChunkedRawDataNodeKey> DeserializeWithTermInfo(Deserializer& de, ChunkedKeyTerminatorInfo term_info);

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

    void Serialize(Serializer& s) const;

    static hdf5::expected<BTreeNode> DeserializeGroup(Deserializer& de);
    static hdf5::expected<BTreeNode> DeserializeChunked(Deserializer& de, ChunkedKeyTerminatorInfo term_info);

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

    hdf5::expected<BTreeNode> ReadChild(Deserializer& de) const;

    [[nodiscard]] BTreeNode Split(KValues k);

    hdf5::expected<cstd::optional<SplitResult>> InsertGroup(offset_t this_offset, offset_t name_offset, offset_t obj_header_ptr, FileLink& file, LocalHeap& heap);

    hdf5::expected<cstd::optional<SplitResultChunked>> InsertChunked(
        offset_t this_offset,
        const BTreeChunkedRawDataNodeKey& key,
        offset_t data_ptr,
        FileLink& file
    );

    hdf5::expected<cstd::optional<uint16_t>> FindGroupIndex(hdf5::string_view key, const LocalHeap& heap, Deserializer& de) const;

    [[nodiscard]] cstd::optional<uint16_t> FindChunkedIndex(const ChunkCoordinates& chunk_coords) const;

    [[nodiscard]] bool AtCapacity(KValues k) const;

    hdf5::expected<uint16_t> GroupInsertionPosition(hdf5::string_view key, const LocalHeap& heap, Deserializer& de) const;

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

                visitor(std::move(*name_result), ptr);
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
                    visitor(key, ptr);
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
    explicit GroupBTree(offset_t addr, std::shared_ptr<FileLink> file, const LocalHeap& heap)
        : file_(std::move(file)), heap_(heap), addr_(addr) {}

    [[nodiscard]] hdf5::expected<cstd::optional<offset_t>> Get(hdf5::string_view name) const;

    hdf5::expected<void> InsertGroup(offset_t name_offset, offset_t object_header_ptr);

    [[nodiscard]] hdf5::expected<size_t> Size() const;

    [[nodiscard]] hdf5::expected<std::vector<offset_t>> Elements() const;
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

    [[nodiscard]] hdf5::expected<std::vector<cstd::tuple<ChunkCoordinates, offset_t, len_t>>> Offsets() const;

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