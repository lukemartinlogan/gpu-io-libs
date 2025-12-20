#pragma once


#include "dataset.h"
#include "local_heap.h"
#include "symbol_table.h"
#include "tree.h"
#include "file_link.h"
#include "object.h"
#include "gpu_string.h"

class Group {
public:
    __device__
    static hdf5::expected<Group> New(const Object& object);

    __device__
    [[nodiscard]] hdf5::expected<Dataset> OpenDataset(hdf5::string_view dataset_name) const;

    __device__
    hdf5::expected<Dataset> CreateDataset(
        hdf5::string_view dataset_name,
        const hdf5::dim_vector<len_t>& dimension_sizes,
        const DatatypeMessage& type,
        cstd::optional<hdf5::dim_vector<uint32_t>> chunk_dims = cstd::nullopt,
        cstd::optional<cstd::inplace_vector<byte_t, FillValueMessage::kMaxFillValueSizeBytes>> fill_value = cstd::nullopt
    );

    __device__
    [[nodiscard]] hdf5::expected<Group> OpenGroup(hdf5::string_view group_name) const;

    __device__
    hdf5::expected<Group> CreateGroup(hdf5::string_view name);

    __device__
    [[nodiscard]] hdf5::expected<cstd::optional<Object>> Get(hdf5::string_view name) const;

private:
    __device__
    hdf5::expected<void> Insert(hdf5::string_view name, offset_t object_header_ptr);

    __device__
    hdf5::expected<void> WriteEntryToNewNode(SymbolTableEntry entry);

    // FIXME: get rid of this method
    __device__
    [[nodiscard]] const LocalHeap& GetLocalHeap() const {
        return table_.heap_;
    }

    // FIXME: get rid of this method
    __device__
    LocalHeap& GetLocalHeap() {
        return table_.heap_;
    }

    // FIXME: get rid of this method
    __device__
    [[nodiscard]] hdf5::expected<SymbolTableNode> GetSymbolTableNode() const;

    __device__
    void UpdateBTreePointer();

    __device__
    Group() = default;

    __device__
    Group(Object object, GroupBTree table)
        : object_(std::move(object)), table_(std::move(table)) {}

private:
    Object object_;

    GroupBTree table_{};
};
