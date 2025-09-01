#pragma once
#include <array>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "file_link.h"
#include "local_heap.h"
#include "object.h"
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

    static constexpr len_t kAllocationSize = sizeof(len_t);
};

static_assert(
    BTreeGroupNodeKey::kAllocationSize == sizeof(BTreeGroupNodeKey),
    "no extra fields should be added to key"
);

struct ChunkCoordinates {
    std::vector<uint64_t> coords;

    ChunkCoordinates() = default;

    explicit ChunkCoordinates(const std::vector<uint64_t>& coordinates) : coords(coordinates) {}
    explicit ChunkCoordinates(std::vector<uint64_t>&& coordinates) : coords(std::move(coordinates)) {}

    auto operator<=>(const ChunkCoordinates&) const = default;

    [[nodiscard]] size_t Dimensions() const {
        return coords.size();
    }
};

struct BTreeChunkedRawDataNodeKey {
    // in bytes
    uint32_t chunk_size;
    // bit field indicating which filters have been skipped for this chunk
    uint32_t filter_mask;
    // .size() == number of dimensions
    // extra uint64_t(0) at the end (not stored)
    ChunkCoordinates chunk_offset_in_dataset;

    void Serialize(Serializer& s) const;

    static BTreeChunkedRawDataNodeKey Deserialize(Deserializer& de);

    [[nodiscard]] uint16_t AllocationSize() const {
        // Key size = chunk_size + filter_mask + (dimensions * sizeof(uint64_t))
        // + 1 for the terminating 0, since this is allocation size
        uint16_t dimensions = static_cast<uint16_t>(chunk_offset_in_dataset.Dimensions());
        return sizeof(uint32_t) + sizeof(uint32_t) + (dimensions + 1) * sizeof(uint64_t);
    }
};

template<typename K>
struct BTreeEntries {
    // TODO: enforce child_pointers.size() + 1 == keys.size()
    std::vector<K> keys;
    std::vector<offset_t> child_pointers;

    [[nodiscard]] uint16_t EntriesUsed() const;

    [[nodiscard]] uint16_t KeySize() const;

    static_assert(
        std::is_same_v<K, BTreeGroupNodeKey> || std::is_same_v<K, BTreeChunkedRawDataNodeKey>,
        "Unsupported key type"
    );
};

struct SplitResult;

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
    std::variant<
        BTreeEntries<BTreeGroupNodeKey>,
        BTreeEntries<BTreeChunkedRawDataNodeKey>
    > entries{};

    // max number of children this node points to
    // all nodes have same max degree (max entries used) but
    // most nodes point to less than that
    [[nodiscard]] uint16_t EntriesUsed() const;

    [[nodiscard]] bool IsLeaf() const {
        return level == 0;
    }

    [[nodiscard]] std::optional<offset_t> Get(std::string_view name, FileLink& file, const LocalHeap& heap) const;

    void Serialize(Serializer& s) const;

    static BTreeNode Deserialize(Deserializer& de);

private:
    friend struct BTree;
    friend class Group;

    struct KValues {
        uint16_t leaf;
        uint16_t internal;

        [[nodiscard]] uint16_t Get(bool is_leaf) const {
            return is_leaf ? leaf : internal;
        }
    };

    [[nodiscard]] BTreeNode Split(KValues k);

    std::optional<SplitResult> InsertGroup(offset_t this_offset, offset_t name_offset, offset_t obj_header_ptr, FileLink& file, LocalHeap& heap);

    std::optional<uint16_t> FindGroupIndex(std::string_view key, const LocalHeap& heap, Deserializer& de) const;

    [[nodiscard]] std::optional<uint16_t> FindChunkedIndex(const ChunkCoordinates& chunk_coords) const;

    [[nodiscard]] bool AtCapacity(KValues k) const;

    uint16_t GroupInsertionPosition(std::string_view key, const LocalHeap& heap, Deserializer& de) const;

    [[nodiscard]] uint16_t ChunkedInsertionPosition(const ChunkCoordinates& chunk_coords) const;

    template <class K>
    [[nodiscard]] K GetMaxKey(FileLink& file) const;

    template <class K>
    [[nodiscard]] K GetMinKey() const;

    [[nodiscard]] len_t AllocationSize(KValues k) const;

    len_t WriteNodeGetAllocSize(offset_t offset, FileLink& file, KValues k) const;

    offset_t AllocateAndWrite(FileLink& file, KValues k) const;

    void Recurse(const std::function<void(std::string, offset_t)>& visitor, FileLink& file) const;

private:
    static constexpr uint8_t kGroupNodeTy = 0, kRawDataChunkNodeTy = 1;
    static constexpr std::array<uint8_t, 4> kSignature = { 'T', 'R', 'E', 'E' };
};

struct SplitResult {
    BTreeGroupNodeKey promoted_key;
    offset_t new_node_offset;
};

struct BTree {
    explicit BTree(offset_t addr, std::shared_ptr<FileLink> file, const LocalHeap& heap)
        : file_(std::move(file)), heap_(heap), addr_(addr) {}

    [[nodiscard]] std::optional<offset_t> Get(std::string_view name) const;

    void InsertGroup(offset_t name_offset, offset_t object_header_ptr);

    [[nodiscard]] size_t Size() const;

    [[nodiscard]] std::vector<offset_t> Elements() const;
private:
    friend class Group;

    BTree() = default;

    [[nodiscard]] std::optional<BTreeNode> ReadRoot() const;

private:
    std::shared_ptr<FileLink> file_{};
    LocalHeap heap_{};
    std::optional<offset_t> addr_{};
};