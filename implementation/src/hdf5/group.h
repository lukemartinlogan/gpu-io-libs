#pragma once
#include "dataset.h"
#include "local_heap.h"
#include "object_header.h"
#include "tree.h"

class Group {
public:
    explicit Group(const ObjectHeader& header, Deserializer& de);

    Dataset GetDataset(std::string_view dataset_name) const;

private:
public:
    Deserializer* read_;

    BTreeNode table_;
    LocalHeap local_heap_;
};
