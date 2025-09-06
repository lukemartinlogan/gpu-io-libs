#pragma once

#include <stdexcept>

#include "file_link.h"
#include "object.h"
#include "object_header.h"
#include "tree.h"
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

    void ReadHyperslab(
        std::span<byte_t> buffer,
        const std::vector<uint64_t>& start,
        const std::vector<uint64_t>& count,
        const std::vector<uint64_t>& stride = {},
        const std::vector<uint64_t>& block = {}
    ) const;

    template<typename T>
    std::vector<T> ReadHyperslab(
        const std::vector<uint64_t>& start,
        const std::vector<uint64_t>& count,
        const std::vector<uint64_t>& stride = {},
        const std::vector<uint64_t>& block = {}
    ) const {
        size_t total_elements = TotalElements(count, block);

        std::vector<T> result(total_elements);
        ReadHyperslab(
            std::span(
                reinterpret_cast<byte_t*>(result.data()),
                result.size() * sizeof(T)
            ),
            start,
            count,
            stride,
            block
        );

        return result;
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

    void WriteHyperslab(
        std::span<const byte_t> data,
        const std::vector<uint64_t>& start,
        const std::vector<uint64_t>& count,
        const std::vector<uint64_t>& stride = {},
        const std::vector<uint64_t>& block = {}
    ) const;

    template<typename T>
    void WriteHyperslab(
        std::span<const T> data,
        const std::vector<uint64_t>& start,
        const std::vector<uint64_t>& count,
        const std::vector<uint64_t>& stride = {},
        const std::vector<uint64_t>& block = {}
    ) const {
        WriteHyperslab(
            std::span(
                reinterpret_cast<const byte_t*>(data.data()),
                data.size_bytes()
            ),
            start,
            count,
            stride,
            block
        );
    }

    [[nodiscard]] std::vector<std::tuple<ChunkCoordinates, offset_t, len_t>> RawOffsets() const;

private:
    static size_t TotalElements(const std::vector<uint64_t>& count, const std::vector<uint64_t>& block) {
        size_t total_elements = 1;
        for (size_t i = 0; i < count.size(); ++i) {
            size_t effective_block = block.empty() ? 1 : block[i];
            total_elements *= count[i] * effective_block;
        }

        return total_elements;
    }


private:
    Object object_;

    DataLayoutMessage layout_{};
    DatatypeMessage type_{};
    DataspaceMessage space_{};
};
