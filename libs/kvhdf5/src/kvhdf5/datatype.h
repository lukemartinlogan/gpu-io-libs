#pragma once

#include "../defines.h"
#include "../serde.h"
#include "types.h"
#include <cuda/std/variant>
#include <cuda/std/optional>

namespace kvhdf5 {

struct PrimitiveType {
    enum class Kind : uint8_t {
        Int8, Int16, Int32, Int64,
        Uint8, Uint16, Uint32, Uint64,
        Float32, Float64,
    } kind;

    CROSS_FUN constexpr PrimitiveType(Kind k) : kind(k) {}

    CROSS_FUN uint32_t GetSize() const {
        switch(kind) {
            case Kind::Int8:
            case Kind::Uint8:   return 1;
            case Kind::Int16:
            case Kind::Uint16:  return 2;
            case Kind::Int32:
            case Kind::Uint32:
            case Kind::Float32: return 4;
            case Kind::Int64:
            case Kind::Uint64:
            case Kind::Float64: return 8;
        }

        KVHDF5_ASSERT(false, "PrimitiveType::GetSize: unreachable");
        return 0;
    }

    CROSS_FUN constexpr bool operator==(const PrimitiveType&) const = default;
    CROSS_FUN constexpr bool operator==(Kind k) const { return kind == k; }
};

struct DatatypeRef {
    cstd::variant<PrimitiveType, DatatypeId> data;

    CROSS_FUN constexpr DatatypeRef(PrimitiveType prim) : data(prim) {}
    CROSS_FUN constexpr DatatypeRef(DatatypeId id) : data(id) {}

    CROSS_FUN bool IsPrimitive() const {
        return cstd::holds_alternative<PrimitiveType>(data);
    }

    CROSS_FUN cstd::optional<PrimitiveType> GetPrimitive() const {
        if (IsPrimitive()) {
            return cstd::get<PrimitiveType>(data);
        }
        return cstd::nullopt;
    }

    CROSS_FUN cstd::optional<DatatypeId> GetComplexId() const {
        if (!IsPrimitive()) {
            return cstd::get<DatatypeId>(data);
        }
        return cstd::nullopt;
    }

    CROSS_FUN constexpr bool operator==(const DatatypeRef&) const = default;

    CROSS_FUN bool operator==(PrimitiveType prim) const {
        return GetPrimitive() == prim;
    }

    CROSS_FUN bool operator==(PrimitiveType::Kind k) const {
        return GetPrimitive() == PrimitiveType(k);
    }

    CROSS_FUN bool operator==(DatatypeId id) const {
        return GetComplexId() == id;
    }

    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        uint8_t index = data.index();
        serde::Write(s, index);

        if (index == 0) {
            serde::Write(s, cstd::get<0>(data));
        } else {
            serde::Write(s, cstd::get<1>(data));
        }
    }

    template<serde::Deserializer D>
    CROSS_FUN static DatatypeRef Deserialize(D& d) {
        uint8_t index = serde::Read<uint8_t>(d);

        if (index == 0) {
            PrimitiveType prim = serde::Read<PrimitiveType>(d);
            return DatatypeRef(prim);
        } else {
            DatatypeId id = serde::Read<DatatypeId>(d);
            return DatatypeRef(id);
        }
    }
};


struct ComplexDatatypeDescriptor {
    enum class Kind : uint8_t {
        Compound,
        Array,
    } kind;
    uint32_t element_size;

    CROSS_FUN constexpr bool operator==(const ComplexDatatypeDescriptor&) const = default;
};

} // namespace kvhdf5

KVHDF5_AUTO_SERDE(kvhdf5::PrimitiveType);
KVHDF5_AUTO_SERDE(kvhdf5::ComplexDatatypeDescriptor);
