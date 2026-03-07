#include <catch2/catch_test_macros.hpp>
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/data_structures/priv/vector.h"
#include "../../common/allocator_fixture.h"

using Allocator = hshm::ipc::BuddyAllocator;

template<typename T>
using vector = hshm::priv::vector<T, Allocator>;

using AllocatorFixture = test::AllocatorFixture<Allocator>;

// ============================================================================
// Test 1: Basic alloc/free/alloc cycle
// Verifies that after a vector is freed, a subsequent allocation works and
// returns correct data — confirming the freed region is reusable.
// ============================================================================
TEST_CASE("BuddyAllocator: alloc then free then alloc again", "[buddy_allocator]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    SECTION("Allocate vector, destroy it, allocate again — data is correct") {
        {
            vector<int> vec(fixture.allocator);
            for (int i = 0; i < 100; ++i) {
                vec.push_back(i);
            }
            REQUIRE(vec.size() == 100);
            REQUIRE(vec[0] == 0);
            REQUIRE(vec[99] == 99);
            // Destructor frees memory back to BuddyAllocator
        }

        vector<int> vec2(fixture.allocator);
        for (int i = 0; i < 50; ++i) {
            vec2.push_back(i * 2);
        }
        REQUIRE(vec2.size() == 50);
        REQUIRE(vec2[0] == 0);
        REQUIRE(vec2[49] == 98);
    }
}

// ============================================================================
// Test 2: Deallocation actually reclaims memory (tight-heap proof)
//
// The fixture gives us a 64KB heap. A vector<int> growing to 10000 elements
// needs ~80KB of peak backing storage (exponential doubling). That exceeds the
// heap, so we use 1000 elements (~16KB peak) and run 8 back-to-back rounds.
//
// Without deallocation: round 2 would OOM (heap already full from round 1).
// With deallocation: every round frees before allocating, so all 8 succeed.
//
// The test heap is 64KB. Each round uses at most ~16KB peak. Eight rounds
// in sequence without freeing = 128KB > 64KB → guaranteed OOM if leaking.
// ============================================================================
TEST_CASE("BuddyAllocator: deallocation reclaims memory (tight heap)", "[buddy_allocator]") {
    AllocatorFixture fixture;  // 64KB heap
    REQUIRE(fixture.IsValid());

    SECTION("8 sequential allocations that would OOM without dealloc") {
        for (int round = 0; round < 8; ++round) {
            // Each vector peaks at ~16KB. 8 * 16KB = 128KB > 64KB heap.
            // If BuddyAllocator doesn't free, this aborts before round 4.
            vector<int> vec(fixture.allocator);
            for (int i = 0; i < 1000; ++i) {
                vec.push_back(round * 1000 + i);
            }
            REQUIRE(vec.size() == 1000);
            REQUIRE(vec[0] == round * 1000);
            REQUIRE(vec[999] == round * 1000 + 999);
            // Destructor must free ~16KB back to the allocator here.
        }
    }
}

// ============================================================================
// Test 3: Multiple concurrent vectors freed together
// Allocates 3 vectors simultaneously, frees them all, then allocates again.
// Verifies the allocator coalesces / handles multiple concurrent live blocks.
// ============================================================================
TEST_CASE("BuddyAllocator: multiple alloc/free cycles", "[buddy_allocator]") {
    AllocatorFixture fixture;
    REQUIRE(fixture.IsValid());

    SECTION("Repeated alloc/free cycles do not exhaust memory") {
        for (int cycle = 0; cycle < 5; ++cycle) {
            {
                vector<int> v1(fixture.allocator);
                vector<int> v2(fixture.allocator);
                vector<int> v3(fixture.allocator);

                for (int i = 0; i < 50; ++i) {
                    v1.push_back(i);
                    v2.push_back(i * 10);
                    v3.push_back(i * 100);
                }

                REQUIRE(v1.size() == 50);
                REQUIRE(v2.size() == 50);
                REQUIRE(v3.size() == 50);
                // All three freed here
            }
        }

        // Confirm allocator still works after all the cycles
        vector<int> final_vec(fixture.allocator);
        for (int i = 0; i < 100; ++i) {
            final_vec.push_back(i);
        }
        REQUIRE(final_vec.size() == 100);
    }
}
