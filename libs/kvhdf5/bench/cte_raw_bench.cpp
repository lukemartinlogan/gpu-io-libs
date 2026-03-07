#include "bench_fixture.h"
#include "wrp_cte/core/core_client.h"
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

// Raw CTE benchmarks: measure the underlying CTE PutBlob/GetBlob latency
// directly, without the kvhdf5 serialization layer. Comparing these numbers
// to the Container-level benchmarks shows how much overhead the kvhdf5
// serialization/deserialization adds.

// ============================================================================
// BM_RawCtePut — raw tag.PutBlob at various sizes
// ============================================================================

class RawCtePutFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        EnsureCteRuntime();
        int64_t bytes = state.range(0);
        data_.resize(bytes);
        for (int64_t i = 0; i < bytes; ++i) {
            data_[i] = static_cast<char>(i & 0xFF);
        }
        tag_name_ = bench::UniqueTagName();
        tag_.emplace(tag_name_);
    }
    void TearDown(benchmark::State&) override {
        if (tag_) {
            WRP_CTE_CLIENT->AsyncDelTag(tag_->GetTagId()).Wait();
        }
        tag_.reset();
    }
protected:
    std::string tag_name_;
    std::optional<wrp_cte::core::Tag> tag_;
    std::vector<char> data_;
};

BENCHMARK_DEFINE_F(RawCtePutFixture, BM_RawCtePut)(benchmark::State& state) {
    const char* blob_name = "blob";
    for (auto _ : state) {
        tag_->PutBlob(blob_name, data_.data(), data_.size());
    }
    state.SetBytesProcessed(state.iterations() * state.range(0));
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RawCtePutFixture, BM_RawCtePut)
    ->Arg(24)       // typical metadata key+size prefix
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536);

// ============================================================================
// BM_RawCteGet — raw tag.GetBlob at various sizes
// ============================================================================

class RawCteGetFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        EnsureCteRuntime();
        int64_t bytes = state.range(0);
        data_.resize(bytes);
        for (int64_t i = 0; i < bytes; ++i) {
            data_[i] = static_cast<char>(i & 0xFF);
        }
        out_.resize(bytes);

        tag_name_ = bench::UniqueTagName();
        tag_.emplace(tag_name_);
        tag_->PutBlob("blob", data_.data(), data_.size());
    }
    void TearDown(benchmark::State&) override {
        if (tag_) {
            WRP_CTE_CLIENT->AsyncDelTag(tag_->GetTagId()).Wait();
        }
        tag_.reset();
    }
protected:
    std::string tag_name_;
    std::optional<wrp_cte::core::Tag> tag_;
    std::vector<char> data_;
    std::vector<char> out_;
};

BENCHMARK_DEFINE_F(RawCteGetFixture, BM_RawCteGet)(benchmark::State& state) {
    const char* blob_name = "blob";
    for (auto _ : state) {
        tag_->GetBlob(blob_name, out_.data(), out_.size());
        benchmark::DoNotOptimize(out_.data());
    }
    state.SetBytesProcessed(state.iterations() * state.range(0));
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RawCteGetFixture, BM_RawCteGet)
    ->Arg(24)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536);

// ============================================================================
// BM_RawCteGetBlobSize — raw tag.GetBlobSize latency
// ============================================================================

class RawCteSizeFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State& state) override {
        EnsureCteRuntime();
        tag_name_ = bench::UniqueTagName();
        tag_.emplace(tag_name_);

        bool exists = state.range(0) != 0;
        if (exists) {
            char data[64] = {};
            tag_->PutBlob("blob", data, sizeof(data));
        }
    }
    void TearDown(benchmark::State&) override {
        if (tag_) {
            WRP_CTE_CLIENT->AsyncDelTag(tag_->GetTagId()).Wait();
        }
        tag_.reset();
    }
protected:
    std::string tag_name_;
    std::optional<wrp_cte::core::Tag> tag_;
};

BENCHMARK_DEFINE_F(RawCteSizeFixture, BM_RawCteGetBlobSize)(benchmark::State& state) {
    const char* blob_name = state.range(0) ? "blob" : "nonexistent";
    for (auto _ : state) {
        auto size = tag_->GetBlobSize(blob_name);
        benchmark::DoNotOptimize(size);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RawCteSizeFixture, BM_RawCteGetBlobSize)
    ->Arg(1)   // exists = true
    ->Arg(0);  // exists = false

// ============================================================================
// BM_RawCteTagCreate — measure tag creation overhead
// ============================================================================

class RawCteTagCreateFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State&) override {
        EnsureCteRuntime();
    }
};

BENCHMARK_DEFINE_F(RawCteTagCreateFixture, BM_RawCteTagCreate)(benchmark::State& state) {
    for (auto _ : state) {
        auto name = bench::UniqueTagName();
        wrp_cte::core::Tag tag(name);
        benchmark::DoNotOptimize(tag);
        state.PauseTiming();
        WRP_CTE_CLIENT->AsyncDelTag(tag.GetTagId()).Wait();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RawCteTagCreateFixture, BM_RawCteTagCreate)
    ->Iterations(5);

// ============================================================================
// BM_CteGetBreakdown — time each CTE round-trip inside CteBlobStore::GetBlob
// ============================================================================

class CteGetBreakdownFixture : public benchmark::Fixture {
public:
    void SetUp(benchmark::State&) override {
        EnsureCteRuntime();
        tag_name_ = bench::UniqueTagName();
        tag_.emplace(tag_name_);

        // Build blob in CteBlobStore format: [uint64_t real_size][data bytes]
        constexpr size_t kDataSize = 64;
        uint64_t real_size = kDataSize;
        size_t total = sizeof(real_size) + kDataSize;
        std::vector<char> buffer(total);
        std::memcpy(buffer.data(), &real_size, sizeof(real_size));
        for (size_t i = 0; i < kDataSize; ++i) {
            buffer[sizeof(real_size) + i] = static_cast<char>(i & 0xFF);
        }

        tag_->PutBlob("blob", buffer.data(), buffer.size());

        out_.resize(kDataSize);
    }
    void TearDown(benchmark::State&) override {
        if (tag_) {
            WRP_CTE_CLIENT->AsyncDelTag(tag_->GetTagId()).Wait();
        }
        tag_.reset();
    }
protected:
    std::string tag_name_;
    std::optional<wrp_cte::core::Tag> tag_;
    std::vector<char> out_;
};

BENCHMARK_DEFINE_F(CteGetBreakdownFixture, BM_CteGetBreakdown)(benchmark::State& state) {
    const char* blob_name = "blob";
    constexpr size_t kDataSize = 64;

    double total_getBlobSize_ns = 0;
    double total_getPrefix_ns = 0;
    double total_getData_ns = 0;

    for (auto _ : state) {
        // Step 1: GetBlobSize
        auto t0 = std::chrono::steady_clock::now();
        auto stored_size = tag_->GetBlobSize(blob_name);
        auto t1 = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(stored_size);

        // Step 2: GetBlob — read 8-byte size prefix
        uint64_t real_size;
        auto t2 = std::chrono::steady_clock::now();
        tag_->GetBlob(blob_name, reinterpret_cast<char*>(&real_size),
                      sizeof(real_size), 0);
        auto t3 = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(real_size);

        // Step 3: GetBlob — read actual data
        auto t4 = std::chrono::steady_clock::now();
        tag_->GetBlob(blob_name, out_.data(), kDataSize, sizeof(uint64_t));
        auto t5 = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(out_.data());

        total_getBlobSize_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        total_getPrefix_ns  += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        total_getData_ns    += std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count();
    }

    double n = static_cast<double>(state.iterations());
    state.counters["getBlobSize_ns"] = benchmark::Counter(total_getBlobSize_ns / n,
        benchmark::Counter::kDefaults);
    state.counters["getPrefix_ns"] = benchmark::Counter(total_getPrefix_ns / n,
        benchmark::Counter::kDefaults);
    state.counters["getData_ns"] = benchmark::Counter(total_getData_ns / n,
        benchmark::Counter::kDefaults);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(CteGetBreakdownFixture, BM_CteGetBreakdown);
