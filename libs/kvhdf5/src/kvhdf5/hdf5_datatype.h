#pragma once

#include "datatype.h"
#include "error.h"
#include "utils/gpu_string.h"
#include <cuda/std/inplace_vector>

namespace kvhdf5 {

class Datatype {
    DatatypeRef ref_;

    struct Field {
        gpu_string<255> name;
        size_t offset;
        PrimitiveType type;
    };

    cstd::inplace_vector<Field, 16> fields_{};
    size_t compound_size_ = 0;
    bool is_compound_ = false;

    CROSS_FUN explicit Datatype(DatatypeRef ref) : ref_(ref) {}

public:
    CROSS_FUN static Datatype Int8()    { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int8))); }
    CROSS_FUN static Datatype Int16()   { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int16))); }
    CROSS_FUN static Datatype Int32()   { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int32))); }
    CROSS_FUN static Datatype Int64()   { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Int64))); }
    CROSS_FUN static Datatype Uint8()   { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Uint8))); }
    CROSS_FUN static Datatype Uint16()  { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Uint16))); }
    CROSS_FUN static Datatype Uint32()  { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Uint32))); }
    CROSS_FUN static Datatype Uint64()  { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Uint64))); }
    CROSS_FUN static Datatype Float32() { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float32))); }
    CROSS_FUN static Datatype Float64() { return Datatype(DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float64))); }

    CROSS_FUN static Datatype CreateCompound(size_t total_size) {
        Datatype dt{DatatypeRef{PrimitiveType{PrimitiveType::Kind::Int8}}};
        dt.compound_size_ = total_size;
        dt.is_compound_ = true;
        return dt;
    }

    CROSS_FUN expected<void> InsertField(gpu_string_view name, size_t offset, Datatype member_type) {
        KVHDF5_ASSERT(is_compound_, "InsertField called on non-compound Datatype");

        auto prim = member_type.ref_.GetPrimitive();
        if (!prim.has_value()) {
            return make_error(ErrorCode::InvalidArgument, "member type must be primitive");
        }

        fields_.push_back(Field{gpu_string<255>(name), offset, prim.value()});
        return {};
    }

    CROSS_FUN uint32_t GetSize() const {
        if (is_compound_) {
            return static_cast<uint32_t>(compound_size_);
        }
        auto prim = ref_.GetPrimitive();
        KVHDF5_ASSERT(prim.has_value(), "GetSize: non-compound Datatype must be primitive");
        return prim.value().GetSize();
    }

    CROSS_FUN bool IsPrimitive() const {
        return !is_compound_ && ref_.IsPrimitive();
    }

    CROSS_FUN bool IsCompound() const {
        return is_compound_;
    }

    CROSS_FUN PrimitiveType::Kind GetPrimitiveKind() const {
        KVHDF5_ASSERT(IsPrimitive(), "GetPrimitiveKind: Datatype is not primitive");
        return ref_.GetPrimitive().value().kind;
    }

    CROSS_FUN size_t GetNumFields() const {
        return fields_.size();
    }

    CROSS_FUN DatatypeRef ToRef() const {
        return ref_;
    }
};

} // namespace kvhdf5
