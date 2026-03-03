#include <catch2/catch_test_macros.hpp>
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/data_structures/priv/vector.h"
#include "../../common/allocator_fixture.h"

using Allocator = hshm::ipc::ArenaAllocator<false>;

template<typename T>
using vector = hshm::priv::vector<T, Allocator>;

using AllocatorFixture = test::AllocatorFixture<Allocator>;

TEST_CASE("iowarp vector basic construction", "[integration][iowarp]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    SECTION("Construction creates empty vector") {
        vector<int> vec(fixture.allocator);
        REQUIRE(vec.size() == 0);
        REQUIRE(vec.empty());
    }

    SECTION("Can push_back elements") {
        vector<int> vec(fixture.allocator);

        vec.push_back(10);
        vec.push_back(20);
        vec.push_back(30);

        REQUIRE(vec.size() == 3);
        REQUIRE(vec[0] == 10);
        REQUIRE(vec[1] == 20);
        REQUIRE(vec[2] == 30);
    }

    SECTION("Can clear vector") {
        vector<int> vec(fixture.allocator);

        vec.push_back(1);
        vec.push_back(2);
        REQUIRE(vec.size() == 2);

        vec.clear();
        REQUIRE(vec.size() == 0);
        REQUIRE(vec.empty());
    }
}

TEST_CASE("iowarp vector with custom types", "[integration][iowarp]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    struct Point {
        float x, y, z;
    };

    SECTION("Can store custom POD types") {
        vector<Point> vec(fixture.allocator);

        vec.push_back({1.0f, 2.0f, 3.0f});
        vec.push_back({4.0f, 5.0f, 6.0f});

        REQUIRE(vec.size() == 2);
        REQUIRE(vec[0].x == 1.0f);
        REQUIRE(vec[0].y == 2.0f);
        REQUIRE(vec[0].z == 3.0f);
        REQUIRE(vec[1].x == 4.0f);
    }
}

TEST_CASE("iowarp vector iteration", "[integration][iowarp]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    vector<int> vec(fixture.allocator);

    vec.push_back(100);
    vec.push_back(200);
    vec.push_back(300);

    SECTION("Can iterate with range-based for") {
        int sum = 0;
        for (const auto& val : vec) {
            sum += val;
        }
        REQUIRE(sum == 600);
    }

    SECTION("Can access via iterators") {
        auto it = vec.begin();
        REQUIRE(*it == 100);
        ++it;
        REQUIRE(*it == 200);
        ++it;
        REQUIRE(*it == 300);
        ++it;
        REQUIRE(it == vec.end());
    }
}

TEST_CASE("iowarp vector with arena allocator", "[integration][iowarp]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    SECTION("Vector works with arena allocator") {
        vector<uint64_t> vec(fixture.allocator);

        vec.push_back(12345);
        vec.push_back(67890);

        REQUIRE(vec.size() == 2);
        REQUIRE(vec[0] == 12345);
        REQUIRE(vec[1] == 67890);
    }

    SECTION("Multiple vectors can share allocator") {
        vector<int> vec1(fixture.allocator);
        vector<int> vec2(fixture.allocator);

        vec1.push_back(1);
        vec1.push_back(2);
        vec2.push_back(10);
        vec2.push_back(20);

        REQUIRE(vec1.size() == 2);
        REQUIRE(vec2.size() == 2);
        REQUIRE(vec1[0] == 1);
        REQUIRE(vec2[0] == 10);
    }
}
