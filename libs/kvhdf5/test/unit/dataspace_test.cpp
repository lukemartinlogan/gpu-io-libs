#include <catch2/catch_test_macros.hpp>
#include <kvhdf5/dataspace.h>

using namespace kvhdf5;

TEST_CASE("CreateSimple 1D", "[dataspace]") {
    uint64_t dims[] = {100};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1));
    REQUIRE(ds.has_value());
    REQUIRE(ds->GetNDims() == 1);
    REQUIRE(ds->GetTotalElements() == 100);
}

TEST_CASE("CreateSimple 2D", "[dataspace]") {
    uint64_t dims[] = {10, 20};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2));
    REQUIRE(ds.has_value());
    REQUIRE(ds->GetNDims() == 2);
    REQUIRE(ds->GetTotalElements() == 200);
}

TEST_CASE("CreateSimple with max_dims", "[dataspace]") {
    uint64_t dims[] = {10, 20};
    uint64_t max[] = {100, 200};
    auto ds = Dataspace::CreateSimple(
        cstd::span<const uint64_t>(dims, 2),
        cstd::span<const uint64_t>(max, 2));
    REQUIRE(ds.has_value());
    uint64_t out_dims[2], out_max[2];
    ds->GetDims(cstd::span<uint64_t>(out_dims, 2), cstd::span<uint64_t>(out_max, 2));
    REQUIRE(out_dims[0] == 10);
    REQUIRE(out_dims[1] == 20);
    REQUIRE(out_max[0] == 100);
    REQUIRE(out_max[1] == 200);
}

TEST_CASE("CreateSimple rejects rank > MAX_DIMS", "[dataspace]") {
    uint64_t dims[MAX_DIMS + 1] = {};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, MAX_DIMS + 1));
    REQUIRE(!ds.has_value());
}

TEST_CASE("CreateScalar", "[dataspace]") {
    auto ds = Dataspace::CreateScalar();
    REQUIRE(ds.GetNDims() == 0);
    REQUIRE(ds.GetTotalElements() == 1);
}

TEST_CASE("GetDims without max_dims", "[dataspace]") {
    uint64_t dims[] = {5, 10};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    uint64_t out[2];
    ds.GetDims(cstd::span<uint64_t>(out, 2));
    REQUIRE(out[0] == 5);
    REQUIRE(out[1] == 10);
}

TEST_CASE("SelectAll is default", "[dataspace][selection]") {
    uint64_t dims[] = {10};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    REQUIRE(ds.GetSelectionType() == SelectionType::All);
    REQUIRE(ds.GetSelectedPointCount() == 10);
}

TEST_CASE("SelectNone", "[dataspace][selection]") {
    uint64_t dims[] = {10};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    ds.SelectNone();
    REQUIRE(ds.GetSelectionType() == SelectionType::None);
    REQUIRE(ds.GetSelectedPointCount() == 0);
}

TEST_CASE("SelectAll after SelectNone", "[dataspace][selection]") {
    uint64_t dims[] = {10};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    ds.SelectNone();
    ds.SelectAll();
    REQUIRE(ds.GetSelectionType() == SelectionType::All);
    REQUIRE(ds.GetSelectedPointCount() == 10);
}

TEST_CASE("SelectHyperslab 1D contiguous block", "[dataspace][selection]") {
    uint64_t dims[] = {100};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    uint64_t start[] = {10}, stride[] = {1}, count[] = {20}, block[] = {1};
    auto err = ds.SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1),
        cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1),
        cstd::span<const uint64_t>(block, 1));
    REQUIRE(err.has_value());
    REQUIRE(ds.GetSelectionType() == SelectionType::Hyperslab);
    REQUIRE(ds.GetSelectedPointCount() == 20);
}

TEST_CASE("SelectHyperslab 2D strided", "[dataspace][selection]") {
    uint64_t dims[] = {100, 200};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 2)).value();
    uint64_t start[] = {5, 10}, stride[] = {2, 3}, count[] = {10, 20}, block[] = {1, 1};
    auto err = ds.SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2),
        cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2),
        cstd::span<const uint64_t>(block, 2));
    REQUIRE(err.has_value());
    REQUIRE(ds.GetSelectedPointCount() == 200);
}

TEST_CASE("SelectHyperslab with blocks", "[dataspace][selection]") {
    uint64_t dims[] = {100};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    uint64_t start[] = {0}, stride[] = {10}, count[] = {3}, block[] = {4};
    auto err = ds.SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1),
        cstd::span<const uint64_t>(stride, 1),
        cstd::span<const uint64_t>(count, 1),
        cstd::span<const uint64_t>(block, 1));
    REQUIRE(err.has_value());
    REQUIRE(ds.GetSelectedPointCount() == 12);
}

TEST_CASE("SelectHyperslab empty stride defaults to 1", "[dataspace][selection]") {
    uint64_t dims[] = {100};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    uint64_t start[] = {0}, count[] = {10};
    auto err = ds.SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start, 1),
        {},
        cstd::span<const uint64_t>(count, 1),
        {});
    REQUIRE(err.has_value());
    REQUIRE(ds.GetSelectedPointCount() == 10);
}

TEST_CASE("SelectHyperslab rejects mismatched rank", "[dataspace][selection]") {
    uint64_t dims[] = {100};
    auto ds = Dataspace::CreateSimple(cstd::span<const uint64_t>(dims, 1)).value();
    uint64_t start[] = {0, 0}, stride[] = {1, 1}, count[] = {10, 10}, block[] = {1, 1};
    auto err = ds.SelectHyperslab(
        SelectionOp::Set,
        cstd::span<const uint64_t>(start, 2),
        cstd::span<const uint64_t>(stride, 2),
        cstd::span<const uint64_t>(count, 2),
        cstd::span<const uint64_t>(block, 2));
    REQUIRE(!err.has_value());
}
