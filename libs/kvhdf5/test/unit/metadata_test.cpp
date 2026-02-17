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
        Attribute original;
        original.name = gpu_string<255>("test_attr");
        original.datatype = DatatypeRef(PrimitiveType::Kind::Float32);
        original.value.push_back(byte_t{0x01});
        original.value.push_back(byte_t{0x02});
        original.value.push_back(byte_t{0x03});
        original.value.push_back(byte_t{0x04});

        original.Serialize(rw);

        rw.Reset();
        auto result = Attribute::Deserialize(rw);

        REQUIRE(result == original);
        REQUIRE(result.name == original.name);
        REQUIRE(result.datatype == original.datatype);
        REQUIRE(result.value.size() == 4);
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

        DatasetMetadata original{
            .id = DatasetId(42),
            .datatype = DatatypeRef(PrimitiveType::Kind::Int32),
            .shape = shape_result.value(),
            .attributes = vector<Attribute>(fixture.allocator),
        };

        // Add attributes
        original.attributes = vector<Attribute>(fixture.allocator);

        Attribute attr1;
        attr1.name = gpu_string<255>("attr1");
        attr1.datatype = DatatypeRef(PrimitiveType::Kind::Float64);
        // Add 8 bytes of dummy data
        for (int i = 0; i < 8; ++i) {
            attr1.value.push_back(byte_t{0});
        }
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

        DatasetMetadata original{
            .id = DatasetId(100),
            .datatype = DatatypeRef(PrimitiveType::Kind::Uint64),
            .shape = shape_result.value(),
            .attributes = vector<Attribute>(fixture.allocator),
        };

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
        GroupEntry original(ChildKind::Dataset, ObjectId(123), gpu_string<255>("child_dataset"));

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupEntry::Deserialize(rw);

        REQUIRE(result == original);
        REQUIRE(result.kind == ChildKind::Dataset);
        REQUIRE(result.object_id == ObjectId(123));
        REQUIRE(result.name == gpu_string<255>("child_dataset"));
    }

    SECTION("GroupEntry with Group kind") {
        GroupEntry original(ChildKind::Group, ObjectId(456), gpu_string<255>("child_group"));

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
        GroupMetadata original{
            .id = GroupId(1),
            .children = vector<GroupEntry>(fixture.allocator),
            .attributes = vector<Attribute>(fixture.allocator),
        };

        // Add children
        original.children.push_back(GroupEntry(ChildKind::Group, ObjectId(10), gpu_string<255>("subgroup1")));
        original.children.push_back(GroupEntry(ChildKind::Dataset, ObjectId(20), gpu_string<255>("dataset1")));

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.children.size() == 2);
        REQUIRE(result.children[0].name == gpu_string<255>("subgroup1"));
        REQUIRE(result.children[1].kind == ChildKind::Dataset);
    }

    SECTION("GroupMetadata with attributes") {
        GroupMetadata original{
            .id = GroupId(2),
            .children = vector<GroupEntry>(fixture.allocator),
            .attributes = vector<Attribute>(fixture.allocator),
        };

        Attribute attr;
        attr.name = gpu_string<255>("group_attr");
        attr.datatype = DatatypeRef(PrimitiveType::Kind::Int8);
        attr.value.push_back(byte_t{42});
        original.attributes.push_back(attr);

        original.Serialize(rw);

        rw.Reset();
        auto result = GroupMetadata::Deserialize(rw, ctx);

        REQUIRE(result.id == original.id);
        REQUIRE(result.attributes.size() == 1);
        REQUIRE(static_cast<int>(result.attributes[0].value[0]) == 42);
    }

    SECTION("Empty GroupMetadata") {
        GroupMetadata original{
            .id = GroupId(3),
            .children = vector<GroupEntry>(fixture.allocator),
            .attributes = vector<Attribute>(fixture.allocator),
        };

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
