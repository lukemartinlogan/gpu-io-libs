#pragma once

#include <optional>

#include "dataset.h"
#include "local_heap.h"
#include "symbol_table.h"
#include "tree.h"
#include "file_link.h"
#include "object.h"

class Group {
public:
    explicit Group(const Object& object);

    [[nodiscard]] Dataset OpenDataset(std::string_view dataset_name) const;

    Dataset CreateDataset(
        std::string_view dataset_name,
        const std::vector<len_t>& dimension_sizes,
        const DatatypeMessage& type,
        std::optional<std::vector<byte_t>> fill_value = std::nullopt
    );

    [[nodiscard]] Group OpenGroup(std::string_view group_name) const;

    Group CreateGroup(std::string_view name);

    [[nodiscard]] std::optional<Object> Get(std::string_view name) const;

private:
    Group() = default;

    void Insert(std::string_view name, offset_t object_header_ptr);

    // FIXME: get rid of this method
    [[nodiscard]] const LocalHeap& GetLocalHeap() const {
        return table_.heap_;
    }

    // FIXME: get rid of this method
    LocalHeap& GetLocalHeap() {
        return table_.heap_;
    }

    // FIXME: get rid of this method
    [[nodiscard]] SymbolTableNode GetSymbolTableNode() const;

    void UpdateBTreePointer();

private:
public:
    Object object_;

    BTree table_{};
};
