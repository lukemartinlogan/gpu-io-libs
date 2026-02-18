#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/dataset.h>
#include <kvhdf5/group.h>
#include <kvhdf5/context.h>
#include <utils/buffer.h>
#include <utils/gpu_string.h>
#include <cuda/std/array>
#include "../common/allocator_fixture.h"

using namespace kvhdf5;
using serde::BufferReaderWriter;
using AllocatorFixture = test::AllocatorFixture<AllocatorImpl>;

TEST_CASE("DatasetShape Create factory method", "[metadata]") {
    SECTION("Valid creation") {
        auto result = DatasetShape::Create({100, 200, 300}, {10, 20, 30});
        REQUIRE(result.has_value());

        auto shape = result.value();
        REQUIRE(shape.Ndims() == 3);
        REQUIRE(shape.Dims()[0] == 100);
        REQUIRE(shape.Dims()[1] == 200);
        REQUIRE(shape.Dims()[2] == 300);
        REQUIRE(shape.ChunkDims()[0] == 10);
        REQUIRE(shape.ChunkDims()[1] == 20);
        REQUIRE(shape.ChunkDims()[2] == 30);
    }

    SECTION("Mismatched sizes") {
        auto result = DatasetShape::Create({100, 200}, {10});
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code == ErrorCode::InvalidArgument);
    }

    SECTION("Zero chunk dimension") {
        auto result = DatasetShape::Create({100, 200}, {10, 0});
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code == ErrorCode::InvalidArgument);
    }

    SECTION("Chunk larger than dim") {
        auto result = DatasetShape::Create({100, 200}, {150, 20});
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code == ErrorCode::InvalidArgument);
    }

    SECTION("Empty dimensions") {
        auto result = DatasetShape::Create({}, {});
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code == ErrorCode::InvalidArgument);
    }

    SECTION("Too many dimensions") {
        auto result = DatasetShape::Create({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code == ErrorCode::InvalidArgument);
    }

    SECTION("Single chunk shape") {
        auto result = DatasetShape::Create({100, 200}, {100, 200});
        REQUIRE(result.has_value());
        REQUIRE(result.value().IsSingleChunk());
    }

    SECTION("Non-single chunk shape") {
        auto result = DatasetShape::Create({100, 200}, {10, 200});
        REQUIRE(result.has_value());
        REQUIRE_FALSE(result.value().IsSingleChunk());
    }
}

TEST_CASE("DatasetShape construction and accessors", "[metadata]") {
    SECTION("Ndims accessor") {
        DatasetShape shape{};
        shape.ndims_ = 3;
        REQUIRE(shape.Ndims() == 3);
    }

    SECTION("Dims span accessor returns correct size") {
        DatasetShape shape{};
        shape.ndims_ = 3;
        shape.dims[0] = 10;
        shape.dims[1] = 20;
        shape.dims[2] = 30;

        auto dims_span = shape.Dims();
        REQUIRE(dims_span.size() == 3);
        REQUIRE(dims_span[0] == 10);
        REQUIRE(dims_span[1] == 20);
        REQUIRE(dims_span[2] == 30);
    }

    SECTION("ChunkDims span accessor returns correct size") {
        DatasetShape shape{};
        shape.ndims_ = 2;
        shape.chunk_dims[0] = 5;
        shape.chunk_dims[1] = 10;

        auto chunk_span = shape.ChunkDims();
        REQUIRE(chunk_span.size() == 2);
        REQUIRE(chunk_span[0] == 5);
        REQUIRE(chunk_span[1] == 10);
    }

    SECTION("Const accessors work") {
        DatasetShape shape{};
        shape.dims = {100, 200};
        shape.chunk_dims = {10, 20};
        shape.ndims_ = 2;

        auto dims = shape.Dims();
        auto chunks = shape.ChunkDims();

        REQUIRE(dims.size() == 2);
        REQUIRE(chunks.size() == 2);
        REQUIRE(dims[0] == 100);
        REQUIRE(chunks[0] == 10);
    }
}

TEST_CASE("DatasetShape serialization", "[metadata]") {
    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize DatasetShape") {
        DatasetShape original{};
        original.ndims_ = 3;
        original.dims[0] = 100;
        original.dims[1] = 200;
        original.dims[2] = 300;
        original.chunk_dims[0] = 10;
        original.chunk_dims[1] = 20;
        original.chunk_dims[2] = 30;

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<DatasetShape>(rw);

        REQUIRE(result == original);
        REQUIRE(result.Ndims() == 3);
        REQUIRE(result.Dims()[0] == 100);
        REQUIRE(result.ChunkDims()[0] == 10);
        REQUIRE_FALSE(result.IsSingleChunk());
    }
}

TEST_CASE("Attribute construction and serialization", "[metadata]") {
    cstd::array<byte_t, 1024> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize Attribute") {
        float test_value = 1.5f;
        Attribute original(
            "test_attr",
            DatatypeRef(PrimitiveType::Kind::Float32),
            test_value
        );

        original.Serialize(rw);

        rw.Reset();
        auto result = Attribute::Deserialize(rw);

        REQUIRE(result == original);
        REQUIRE(result.name == original.name);
        REQUIRE(result.datatype == original.datatype);
        REQUIRE(result.value.size() == 4);
    }

    SECTION("POD constructor and Get method") {
        // Test with int32_t
        int32_t int_value = 42;
        Attribute int_attr("int_attr", DatatypeRef(PrimitiveType::Kind::Int32), int_value);
        REQUIRE(int_attr.value.size() == sizeof(int32_t));
        REQUIRE(int_attr.Get<int32_t>() == 42);

        // Test with float
        float float_value = 3.14f;
        Attribute float_attr("float_attr", DatatypeRef(PrimitiveType::Kind::Float32), float_value);
        REQUIRE(float_attr.value.size() == sizeof(float));
        REQUIRE(float_attr.Get<float>() == 3.14f);

        // Test with double
        double double_value = 2.71828;
        Attribute double_attr("double_attr", DatatypeRef(PrimitiveType::Kind::Float64), double_value);
        REQUIRE(double_attr.value.size() == sizeof(double));
        REQUIRE(double_attr.Get<double>() == 2.71828);

        // Test with uint8_t
        uint8_t uint8_value = 255;
        Attribute uint8_attr("uint8_attr", DatatypeRef(PrimitiveType::Kind::Uint8), uint8_value);
        REQUIRE(uint8_attr.value.size() == sizeof(uint8_t));
        REQUIRE(uint8_attr.Get<uint8_t>() == 255);

        // Test serialization round-trip with POD constructor
        int_attr.Serialize(rw);
        rw.Reset();
        auto deserialized = Attribute::Deserialize(rw);
        REQUIRE(deserialized.Get<int32_t>() == 42);
    }
}

TEST_CASE("DatasetMetadata with allocator", "[metadata]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    Context ctx(fixture.allocator);

    cstd::array<byte_t, 2048> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize DatasetMetadata with attributes") {
        auto shape_result = DatasetShape::Create({100, 200}, {10, 20});
        REQUIRE(shape_result.has_value());

        DatasetMetadata original(
            DatasetId(42),
            DatatypeRef(PrimitiveType::Kind::Int32),
            shape_result.value(),
            vector<Attribute>(fixture.allocator)
        );

        // Add attributes
        original.attributes = vector<Attribute>(fixture.allocator);

        double test_value = 3.14159;
        Attribute attr1(
            "attr1",
            DatatypeRef(PrimitiveType::Kind::Float64),
            test_value
        );
        original.attributes.push_back(attr1);

        original.Serialize(rw);

        rw.Reset();
        auto result = DatasetMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.datatype == original.datatype);
        REQUIRE(result.shape == original.shape);
        REQUIRE(result.attributes.size() == 1);
        REQUIRE(result.attributes[0].name == attr1.name);
    }

    SECTION("DatasetMetadata with no attributes") {
        auto shape_result = DatasetShape::Create({1000}, {100});
        REQUIRE(shape_result.has_value());

        DatasetMetadata original(
            DatasetId(100),
            DatatypeRef(PrimitiveType::Kind::Uint64),
            shape_result.value(),
            vector<Attribute>(fixture.allocator)
        );

        original.Serialize(rw);

        rw.Reset();
        auto result = DatasetMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.attributes.size() == 0);
    }
}

TEST_CASE("GroupEntry construction and serialization", "[metadata]") {
    cstd::array<byte_t, 1024> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize GroupEntry") {
        GroupEntry original = GroupEntry::NewDataset(DatasetId(ObjectId(123)), "child_dataset");

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupEntry::Deserialize(rw);

        REQUIRE(result == original);
        REQUIRE(result.kind == ChildKind::Dataset);
        REQUIRE(result.object_id == ObjectId(123));
        REQUIRE(result.name == gpu_string<255>("child_dataset"));
    }

    SECTION("GroupEntry with Group kind") {
        GroupEntry original = GroupEntry::NewGroup(GroupId(ObjectId(456)), "child_group");

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupEntry::Deserialize(rw);

        REQUIRE(result == original);
        REQUIRE(result.kind == ChildKind::Group);
    }
}

TEST_CASE("GroupMetadata with allocator", "[metadata]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());
    Context ctx(fixture.allocator);

    cstd::array<byte_t, 4096> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize GroupMetadata with children") {
        GroupMetadata original(
            GroupId(1),
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        );

        // Add children
        original.children.push_back(GroupEntry::NewGroup(GroupId(ObjectId(10)), "subgroup1"));
        original.children.push_back(GroupEntry::NewDataset(DatasetId(ObjectId(20)), "dataset1"));

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.children.size() == 2);
        REQUIRE(result.children[0].name == gpu_string<255>("subgroup1"));
        REQUIRE(result.children[1].kind == ChildKind::Dataset);
    }

    SECTION("GroupMetadata with attributes") {
        GroupMetadata original(
            GroupId(2),
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        );

        int8_t attr_value = 42;
        Attribute attr(
            "group_attr",
            DatatypeRef(PrimitiveType::Kind::Int8),
            attr_value
        );
        original.attributes.push_back(attr);

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.attributes.size() == 1);
        REQUIRE(static_cast<int>(result.attributes[0].value[0]) == 42);
    }

    SECTION("Empty GroupMetadata") {
        GroupMetadata original(
            GroupId(3),
            vector<GroupEntry>(fixture.allocator),
            vector<Attribute>(fixture.allocator)
        );

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.children.size() == 0);
        REQUIRE(result.attributes.size() == 0);
    }
}

TEST_CASE("ChildKind serialization", "[metadata]") {
    // Compile-time checks to debug serde support
    static_assert(serde::IsPOD<ChildKind>, "ChildKind should be POD");
    static_assert(serde::SerializePOD<ChildKind>::value, "ChildKind should be opted into serde");

    cstd::array<byte_t, 256> buffer;
    BufferReaderWriter rw(buffer);

    SECTION("Serialize and deserialize ChildKind") {
        ChildKind original = ChildKind::Dataset;

        serde::Write(rw, original);

        rw.Reset();
        auto result = serde::Read<ChildKind>(rw);

        REQUIRE(result == original);
    }
}
