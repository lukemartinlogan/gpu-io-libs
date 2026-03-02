#include "bench_fixture.h"
#include "wrp_cte/core/core_client.h"
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
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(RawCteTagCreateFixture, BM_RawCteTagCreate)
    ->Iterations(5);
