#pragma once

#include <optional>

#include "dataset.h"
#include "local_heap.h"
#include "object_header.h"
#include "symbol_table.h"
#include "tree.h"

class Group {
public:
    Group() {}

    explicit Group(const ObjectHeader& header, Deserializer& de);

    Dataset GetDataset(std::string_view dataset_name) const;

    Group GetGroup(std::string_view group_name) const;

private:
    SymbolTableNode GetSymbolTableNode() const;

    std::optional<ObjectHeader> GetEntryWithName(std::string_view name) const;

private:
public:
    Deserializer* read_{};

    BTreeNode table_{};
    LocalHeap local_heap_{};
};
