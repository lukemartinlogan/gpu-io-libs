#include "bench_fixture.h"
#include "hermes_shm/data_structures/priv/vector.h"

template<typename T>
using vector = hshm::priv::vector<T, kvhdf5::AllocatorImpl>;

// ============================================================================
// VectorPushBack: push 1000 ints, measure throughput.
// Allocator is reset before each iteration so we measure steady-state cost
// of allocation + push_back with a fresh heap.
// ============================================================================

class VectorPushBackFixture : public bench::AllocatorFixture {};

BENCHMARK_DEFINE_F(VectorPushBackFixture, VectorPushBack)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        state.ResumeTiming();

        vector<int> vec(GetAllocator());
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
        }
        benchmark::DoNotOptimize(vec);
        // vec destructor frees back to BuddyAllocator
    }
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK_REGISTER_F(VectorPushBackFixture, VectorPushBack);

// ============================================================================
// VectorIterate: iterate over a pre-filled 1000-element vector.
// Measures pure traversal cost — no allocation involved.
// ============================================================================

class VectorIterateFixture : public bench::AllocatorFixture {
public:
    void SetUp(benchmark::State& state) override {
        bench::AllocatorFixture::SetUp(state);
        vec_.emplace(GetAllocator());
        for (int i = 0; i < 1000; ++i) {
            vec_->push_back(i);
        }
    }
    void TearDown(benchmark::State& state) override {
        vec_.reset();
        bench::AllocatorFixture::TearDown(state);
    }
protected:
    std::optional<vector<int>> vec_;
};

BENCHMARK_DEFINE_F(VectorIterateFixture, VectorIterate)(benchmark::State& state) {
    for (auto _ : state) {
        int64_t sum = 0;
        for (const auto& val : *vec_) {
            sum += val;
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK_REGISTER_F(VectorIterateFixture, VectorIterate);

// ============================================================================
// VectorAllocFree: push 1000 ints, destroy vector, allocate again — NO reset.
//
// This benchmark validates that BuddyAllocator actually reclaims memory.
// The fixture uses a 4MB heap. Each 1000-int vector needs ~16KB peak (due to
// exponential growth doubling). Without real deallocation the heap would be
// exhausted within ~250 iterations; with deallocation it runs indefinitely.
// Google Benchmark runs thousands of iterations, so an OOM crash here means
// deallocation is broken.
// ============================================================================

class VectorAllocFreeFixture : public bench::AllocatorFixture {
public:
    void SetUp(benchmark::State& state) override {
        heap_size_ = 4ULL * 1024 * 1024;  // 4MB — tight enough to OOM fast if leak
        bench::AllocatorFixture::SetUp(state);
    }
};

BENCHMARK_DEFINE_F(VectorAllocFreeFixture, VectorAllocFree)(benchmark::State& state) {
    for (auto _ : state) {
        {
            vector<int> vec(GetAllocator());
            for (int i = 0; i < 1000; ++i) {
                vec.push_back(i);
            }
            benchmark::DoNotOptimize(vec);
            // vec destructor: BuddyAllocator must free memory here.
            // If it doesn't, the heap fills up and the next iteration aborts.
        }
        // Immediately allocate again to confirm the freed block is reusable.
        vector<int> probe(GetAllocator());
        probe.push_back(42);
        benchmark::DoNotOptimize(probe);
    }
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK_REGISTER_F(VectorAllocFreeFixture, VectorAllocFree);
