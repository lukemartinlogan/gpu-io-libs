#pragma once

#include <stdexcept>

#include "object_header.h"
#include "../serialization/serialization.h"

class Dataset {
public:
    explicit Dataset(const ObjectHeader& header, /* temporary */ ReaderWriter& file);

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

        file_.SetPosition(/* superblock.base_addr + */ props->address + index * size);

        return file_.ReadComplex<T>();
    }

    // FIXME: implement datatype
    void Read(std::span<byte_t> buffer, size_t start_index, size_t count) const;

    // FIXME: impl datatype instead
    template<typename T>
    std::vector<T> Read(size_t start_index, size_t count) const {
        std::vector<T> out(count);

        Read(
            std::span(
                reinterpret_cast<byte_t*>(out.data()),
                out.size() * sizeof(T)
            ),
            start_index,
            count
        );

        return out;
    }

    void Write(std::span<const byte_t> data, size_t start_index) const;

    template<typename T>
    void Write(std::span<const T> data, size_t start_index) const {
        Write(
            std::span(
                reinterpret_cast<const byte_t*>(data.data()),
                data.size_bytes()
            ),
            start_index
        );
    }

private:
    // TODO: handle references in a better way
    // TODO: make these nullable
    ReaderWriter& file_;

    DataLayoutMessage layout_{};
    DatatypeMessage type_{};
    DataspaceMessage space_{};
};
