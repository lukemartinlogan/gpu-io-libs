#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/hdf5_datatype.h>

using namespace kvhdf5;

TEST_CASE("Primitive type factories", "[hdf5_datatype]") {
    REQUIRE(Datatype::Int8().GetSize() == 1);
    REQUIRE(Datatype::Int16().GetSize() == 2);
    REQUIRE(Datatype::Int32().GetSize() == 4);
    REQUIRE(Datatype::Int64().GetSize() == 8);
    REQUIRE(Datatype::Uint8().GetSize() == 1);
    REQUIRE(Datatype::Uint16().GetSize() == 2);
    REQUIRE(Datatype::Uint32().GetSize() == 4);
    REQUIRE(Datatype::Uint64().GetSize() == 8);
    REQUIRE(Datatype::Float32().GetSize() == 4);
    REQUIRE(Datatype::Float64().GetSize() == 8);
}

TEST_CASE("Primitive type queries", "[hdf5_datatype]") {
    auto dt = Datatype::Float32();
    REQUIRE(dt.IsPrimitive());
    REQUIRE(!dt.IsCompound());
    REQUIRE(dt.GetPrimitiveKind() == PrimitiveType::Kind::Float32);
}

TEST_CASE("ToRef round-trip", "[hdf5_datatype]") {
    auto dt = Datatype::Int32();
    auto ref = dt.ToRef();
    REQUIRE(ref.IsPrimitive());
    REQUIRE(ref.GetPrimitive().value() == PrimitiveType::Kind::Int32);
}

TEST_CASE("Compound type creation", "[hdf5_datatype]") {
    auto dt = Datatype::CreateCompound(16);
    REQUIRE(dt.IsCompound());
    REQUIRE(!dt.IsPrimitive());
    REQUIRE(dt.GetSize() == 16);
    REQUIRE(dt.GetNumFields() == 0);
}

TEST_CASE("Compound type InsertField", "[hdf5_datatype]") {
    auto dt = Datatype::CreateCompound(12);
    REQUIRE(dt.InsertField("x", 0, Datatype::Float32()).has_value());
    REQUIRE(dt.InsertField("y", 4, Datatype::Float32()).has_value());
    REQUIRE(dt.InsertField("z", 8, Datatype::Float32()).has_value());
    REQUIRE(dt.GetNumFields() == 3);
}
