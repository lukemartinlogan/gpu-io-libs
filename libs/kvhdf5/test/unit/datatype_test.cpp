#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/datatype.h>
#include <utils/buffer.h>
#include <cuda/std/array>

using namespace kvhdf5;
using serde::BufferReaderWriter;

TEST_CASE("PrimitiveType element sizes", "[datatype]") {
    SECTION("8-bit types") {
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Int8).GetSize() == 1);
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Uint8).GetSize() == 1);
    }

    SECTION("16-bit types") {
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Int16).GetSize() == 2);
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Uint16).GetSize() == 2);
    }

    SECTION("32-bit types") {
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Int32).GetSize() == 4);
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Uint32).GetSize() == 4);
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Float32).GetSize() == 4);
    }

    SECTION("64-bit types") {
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Int64).GetSize() == 8);
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Uint64).GetSize() == 8);
        REQUIRE(PrimitiveType(PrimitiveType::Kind::Float64).GetSize() == 8);
    }
}

TEST_CASE("PrimitiveType equality", "[datatype]") {
    PrimitiveType int32a(PrimitiveType::Kind::Int32);
    PrimitiveType int32b(PrimitiveType::Kind::Int32);
    PrimitiveType float32(PrimitiveType::Kind::Float32);

    SECTION("Same types are equal") {
        REQUIRE(int32a == int32b);
    }

    SECTION("Different types are not equal") {
        REQUIRE_FALSE(int32a == float32);
    }

    SECTION("Convenience equality with Kind") {
        REQUIRE(int32a == PrimitiveType::Kind::Int32);
        REQUIRE_FALSE(int32a == PrimitiveType::Kind::Float32);
    }
}

TEST_CASE("DatatypeRef with primitive type", "[datatype]") {
    PrimitiveType prim(PrimitiveType::Kind::Float64);
    DatatypeRef ref(prim);

    SECTION("IsPrimitive returns true") {
        REQUIRE(ref.IsPrimitive());
    }

    SECTION("GetPrimitive returns the primitive type") {
        auto result = ref.GetPrimitive();
        REQUIRE(result.has_value());
        REQUIRE(result.value() == prim);
    }

    SECTION("GetComplexId returns nullopt") {
        auto result = ref.GetComplexId();
        REQUIRE_FALSE(result.has_value());
    }
}

TEST_CASE("DatatypeRef with complex type", "[datatype]") {
    DatatypeId complex_id(123);
    DatatypeRef ref(complex_id);

    SECTION("IsPrimitive returns false") {
        REQUIRE_FALSE(ref.IsPrimitive());
    }

    SECTION("GetPrimitive returns nullopt") {
        auto result = ref.GetPrimitive();
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("GetComplexId returns the datatype ID") {
        auto result = ref.GetComplexId();
        REQUIRE(result.has_value());
        REQUIRE(result.value() == complex_id);
    }
}

TEST_CASE("DatatypeRef equality", "[datatype]") {
    PrimitiveType prim1(PrimitiveType::Kind::Float32);
    PrimitiveType prim2(PrimitiveType::Kind::Float32);
    PrimitiveType prim3(PrimitiveType::Kind::Float64);
    DatatypeId id1(100);
    DatatypeId id2(100);
    DatatypeId id3(200);

    SECTION("Same primitive types are equal") {
        DatatypeRef ref1(prim1);
        DatatypeRef ref2(prim2);
        REQUIRE(ref1 == ref2);
    }

    SECTION("Different primitive types are not equal") {
        DatatypeRef ref1(prim1);
        DatatypeRef ref3(prim3);
        REQUIRE_FALSE(ref1 == ref3);
    }

    SECTION("Same complex IDs are equal") {
        DatatypeRef ref1(id1);
        DatatypeRef ref2(id2);
        REQUIRE(ref1 == ref2);
    }

    SECTION("Different complex IDs are not equal") {
        DatatypeRef ref1(id1);
        DatatypeRef ref3(id3);
        REQUIRE_FALSE(ref1 == ref3);
    }

    SECTION("Primitive and complex are not equal") {
        DatatypeRef ref_prim(prim1);
        DatatypeRef ref_complex(id1);
        REQUIRE_FALSE(ref_prim == ref_complex);
    }

    SECTION("Convenience equality with PrimitiveType") {
        DatatypeRef ref(prim1);
        REQUIRE(ref == prim1);
        REQUIRE_FALSE(ref == prim3);
    }

    SECTION("Convenience equality with PrimitiveType::Kind") {
        DatatypeRef ref(prim1);
        REQUIRE(ref == PrimitiveType::Kind::Float32);
        REQUIRE_FALSE(ref == PrimitiveType::Kind::Float64);
    }

    SECTION("Convenience equality with DatatypeId") {
        DatatypeRef ref(id1);
        REQUIRE(ref == id1);
        REQUIRE_FALSE(ref == id3);
    }

    SECTION("Convenience operators don't match across categories") {
        DatatypeRef ref_prim(prim1);
        DatatypeRef ref_complex(id1);
        REQUIRE_FALSE(ref_prim == id1);
        REQUIRE_FALSE(ref_complex == prim1);
        REQUIRE_FALSE(ref_complex == PrimitiveType::Kind::Float32);
    }
}

TEST_CASE("PrimitiveType serialization", "[datatype]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize primitive type") {
        PrimitiveType original(PrimitiveType::Kind::Uint64);

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<PrimitiveType>(rw);
        REQUIRE(result == original);
        REQUIRE(result.GetSize() == 8);
    }
}

TEST_CASE("DatatypeRef serialization with primitive", "[datatype]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize primitive DatatypeRef") {
        PrimitiveType prim(PrimitiveType::Kind::Int16);
        DatatypeRef original(prim);

        original.Serialize(rw);

        rw.Reset();
        DatatypeRef result = DatatypeRef::Deserialize(rw);
        REQUIRE(result == original);
        REQUIRE(result.IsPrimitive());
        REQUIRE(result.GetPrimitive().value().kind == PrimitiveType::Kind::Int16);
    }
}

TEST_CASE("DatatypeRef serialization with complex", "[datatype]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize complex DatatypeRef") {
        DatatypeId complex_id(999);
        DatatypeRef original(complex_id);

        original.Serialize(rw);

        rw.Reset();
        DatatypeRef result = DatatypeRef::Deserialize(rw);
        REQUIRE(result == original);
        REQUIRE_FALSE(result.IsPrimitive());
        REQUIRE(result.GetComplexId().value() == complex_id);
    }
}

TEST_CASE("ComplexDatatypeDescriptor construction", "[datatype]") {
    SECTION("Aggregate initialization") {
        ComplexDatatypeDescriptor desc{ComplexDatatypeDescriptor::Kind::Array, 24};
        REQUIRE(desc.kind == ComplexDatatypeDescriptor::Kind::Array);
        REQUIRE(desc.element_size == 24);
    }
}

TEST_CASE("ComplexDatatypeDescriptor equality", "[datatype]") {
    ComplexDatatypeDescriptor desc1{ComplexDatatypeDescriptor::Kind::Compound, 16};
    ComplexDatatypeDescriptor desc2{ComplexDatatypeDescriptor::Kind::Compound, 16};
    ComplexDatatypeDescriptor desc3{ComplexDatatypeDescriptor::Kind::Array, 16};
    ComplexDatatypeDescriptor desc4{ComplexDatatypeDescriptor::Kind::Compound, 32};

    SECTION("Same kind and size are equal") {
        REQUIRE(desc1 == desc2);
    }

    SECTION("Different kind are not equal") {
        REQUIRE_FALSE(desc1 == desc3);
    }

    SECTION("Different size are not equal") {
        REQUIRE_FALSE(desc1 == desc4);
    }
}

TEST_CASE("ComplexDatatypeDescriptor serialization", "[datatype]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize descriptor") {
        ComplexDatatypeDescriptor original{ComplexDatatypeDescriptor::Kind::Array, 128};

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<ComplexDatatypeDescriptor>(rw);
        REQUIRE(result == original);
    }
}
