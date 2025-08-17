#pragma once

#include <stdexcept>

#include "file_link.h"
#include "object.h"
#include "object_header.h"
#include "../serialization/serialization.h"

class Dataset {
public:
    explicit Dataset(const Object& object);

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

        object_.file->io.SetPosition(object_.file->superblock.base_addr + props->address + index * size);

        return object_.file->io.ReadComplex<T>();
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
    Object object_;

    DataLayoutMessage layout_{};
    DatatypeMessage type_{};
    DataspaceMessage space_{};
};
