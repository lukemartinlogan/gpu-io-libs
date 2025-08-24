#pragma once
#include <array>
#include <functional>
#include <optional>
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

    [[nodiscard]] uint16_t EntriesUsed() const;
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

    struct KValues {
        uint16_t leaf;
        uint16_t internal;

        [[nodiscard]] uint16_t Get(bool is_leaf) const {
            return is_leaf ? leaf : internal;
        }
    };

    [[nodiscard]] BTreeNode Split(KValues k) const;

    std::optional<SplitResult> Insert(offset_t this_offset, offset_t name_offset, offset_t obj_header_ptr, FileLink& file, LocalHeap& heap);

    std::optional<uint16_t> FindIndex(std::string_view key, const LocalHeap& heap, Deserializer& de) const;

    [[nodiscard]] bool AtCapacity(KValues k) const;

    uint16_t InsertionPosition(std::string_view key, const LocalHeap& heap, Deserializer& de) const;

    [[nodiscard]] BTreeGroupNodeKey GetMaxKey(FileLink& file) const;

    [[nodiscard]] BTreeGroupNodeKey GetMinKey() const;

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

    void Insert(offset_t name_offset, offset_t object_header_ptr);
    void Insert(const std::string& name, offset_t object_header_ptr);

    [[nodiscard]] size_t Size() const;

private:
    friend class Group;

    BTree() = default;

    [[nodiscard]] std::optional<BTreeNode> ReadRoot() const;

private:
    std::shared_ptr<FileLink> file_{};
    LocalHeap heap_{};
    std::optional<offset_t> addr_{};
};