#pragma once

#include <optional>

#include "dataset.h"
#include "local_heap.h"
#include "object_header.h"
#include "symbol_table.h"
#include "tree.h"
#include "file_link.h"

class Group {
public:
    explicit Group(const ObjectHeader& header, const std::shared_ptr<FileLink>& file);

    [[nodiscard]] Dataset GetDataset(std::string_view dataset_name) const;

    [[nodiscard]] Group OpenGroup(std::string_view group_name) const;

private:
    Group() = default;

    [[nodiscard]] SymbolTableNode GetSymbolTableNode() const;

    [[nodiscard]] std::optional<ObjectHeader> GetEntryWithName(std::string_view name) const;

private:
public:
    std::shared_ptr<FileLink> file_;

    BTreeNode table_{};
    LocalHeap local_heap_{};
};
