#pragma once
#include "local_heap.h"
#include "object_header.h"
#include "superblock.h"
#include "tree.h"
#include "../serialization/stdio.h"

class File;

class Dataset {
public:
    explicit Dataset(const ObjectHeader& header, /* temporary */ Deserializer& de);

    // TODO: multidimensional coords
    template <typename T>
    T Get(size_t index) const {
        if (index >= space_.TotalElements()) {
            throw std::out_of_range("Index out of bounds for dataset");
        }

        if (type_.class_v == DatatypeMessage::Class::kVariableLength) {
            throw std::logic_error("Variable length datatypes are not supported yet");
        }

        const auto* props = std::get_if<ContiguousStorageProperty>(&layout_.properties);

        if (!props) {
            throw std::logic_error("only contiguous storage currently supported");
        }

        size_t size = type_.Size();

        read_.SetPosition(/* superblock.base_addr + */ props->address + index * size);

        return read_.ReadComplex<T>();
    }

private:
    // TODO: handle references in a better way
    Deserializer& read_;

    DataLayoutMessage layout_{};
    DatatypeMessage type_{};
    DataspaceMessage space_{};
};

class File {
public:
    explicit File(const std::filesystem::path& path);

    Dataset GetDataset(std::string_view dataset_name);

private:
    StdioReader read_;
    SuperblockV0 superblock_;

    // root group
    BTreeNode group_table_;
    LocalHeap local_heap_;
};
