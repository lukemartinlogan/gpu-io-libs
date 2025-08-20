#pragma once

#include <optional>

#include "dataset.h"
#include "local_heap.h"
#include "object_header.h"
#include "symbol_table.h"
#include "tree.h"
#include "file_link.h"
#include "object.h"

class Group {
public:
    explicit Group(const Object& object);

    [[nodiscard]] Dataset GetDataset(std::string_view dataset_name) const;

    [[nodiscard]] Group OpenGroup(std::string_view group_name) const;

    [[nodiscard]] std::optional<Object> Get(std::string_view name) const;

private:
    Group() = default;

    // FIXME: get rid of this method
    [[nodiscard]] SymbolTableNode GetSymbolTableNode() const;

    // FIXME: get rid of this method
    [[nodiscard]] std::optional<Object> GetEntryWithName(std::string_view name) const;

private:
public:
    Object object_;

    BTreeNode table_{};
    LocalHeap local_heap_{};
};
