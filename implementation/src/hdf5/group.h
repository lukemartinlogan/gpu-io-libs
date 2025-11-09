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
    static hdf5::expected<Group> New(const Object& object);

    [[nodiscard]] hdf5::expected<Dataset> OpenDataset(hdf5::string_view dataset_name) const;

    hdf5::expected<Dataset> CreateDataset(
        hdf5::string_view dataset_name,
        const hdf5::dim_vector<len_t>& dimension_sizes,
        const DatatypeMessage& type,
        cstd::optional<hdf5::dim_vector<uint32_t>> chunk_dims = cstd::nullopt,
        cstd::optional<std::vector<byte_t>> fill_value = cstd::nullopt
    );

    [[nodiscard]] hdf5::expected<Group> OpenGroup(hdf5::string_view group_name) const;

    hdf5::expected<Group> CreateGroup(hdf5::string_view name);

    [[nodiscard]] hdf5::expected<cstd::optional<Object>> Get(hdf5::string_view name) const;

private:
    hdf5::expected<void> Insert(hdf5::string_view name, offset_t object_header_ptr);

    // FIXME: get rid of this method
    [[nodiscard]] const LocalHeap& GetLocalHeap() const {
        return table_.heap_;
    }

    // FIXME: get rid of this method
    LocalHeap& GetLocalHeap() {
        return table_.heap_;
    }

    // FIXME: get rid of this method
    [[nodiscard]] hdf5::expected<SymbolTableNode> GetSymbolTableNode() const;

    void UpdateBTreePointer();

    Group() = default;

    Group(Object object, GroupBTree table)
        : object_(std::move(object)), table_(std::move(table)) {}

private:
public:
    Object object_;

    GroupBTree table_{};
};
