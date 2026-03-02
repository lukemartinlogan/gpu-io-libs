#include "bench_fixture.h"
#include "kvhdf5/container.h"
#include "kvhdf5/cte_blob_store.h"
#include <vector>

using namespace kvhdf5;

// ============================================================================
// BM_AllocateId
// ============================================================================

class AllocateIdFixture : public bench::CteFixture {
public:
    void SetUp(benchmark::State& state) override {
        bench::CteFixture::SetUp(state);
        container_.emplace(CreateContainer());
    }
    void TearDown(benchmark::State& state) override {
        container_.reset();
        bench::CteFixture::TearDown(state);
    }
protected:
    std::optional<Container<CteBlobStore>> container_;
};

BENCHMARK_DEFINE_F(AllocateIdFixture, BM_AllocateId)(benchmark::State& state) {
    for (auto _ : state) {
        auto id = container_->AllocateId();
        benchmark::DoNotOptimize(id);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(AllocateIdFixture, BM_AllocateId);

// ============================================================================
// BM_PutGroup / BM_GetGroup
// ============================================================================

class GroupOpFixture : public bench::CteFixture {
public:
    void SetUp(benchmark::State& state) override {
        bench::CteFixture::SetUp(state);
        container_.emplace(CreateContainer());

        num_children_ = state.range(0);
        group_id_ = GroupId(container_->AllocateId());

        // Pre-allocate child IDs (stored in std::vector, not arena)
        for (int64_t i = 0; i < num_children_; ++i) {
            child_ids_.push_back(GroupId(container_->AllocateId()));
        }

        // Create initial group in CTE (for GetGroup benchmarks)
        GroupMetadata meta(group_id_, *GetAllocator());
        for (int64_t i = 0; i < num_children_; ++i) {
            std::string name = "child_" + std::to_string(i);
            meta.children.push_back(
                GroupEntry::NewGroup(child_ids_[i], gpu_string_view(name.c_str())));
        }
        container_->PutGroup(group_id_, meta);
    }
    void TearDown(benchmark::State& state) override {
        child_ids_.clear();
        container_.reset();
        bench::CteFixture::TearDown(state);
    }
protected:
    std::optional<Container<CteBlobStore>> container_;
    GroupId group_id_;
    int64_t num_children_ = 0;
    std::vector<GroupId> child_ids_;
};

BENCHMARK_DEFINE_F(GroupOpFixture, BM_PutGroup)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        GroupMetadata meta(group_id_, *GetAllocator());
        for (int64_t i = 0; i < num_children_; ++i) {
            std::string name = "child_" + std::to_string(i);
            meta.children.push_back(
                GroupEntry::NewGroup(child_ids_[i], gpu_string_view(name.c_str())));
        }
        state.ResumeTiming();
        container_->PutGroup(group_id_, meta);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupOpFixture, BM_PutGroup)
    ->Arg(0)->Arg(5)->Arg(10);

BENCHMARK_DEFINE_F(GroupOpFixture, BM_GetGroup)(benchmark::State& state) {
    for (auto _ : state) {
        ResetAllocator();
        auto result = container_->GetGroup(group_id_);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(GroupOpFixture, BM_GetGroup)
    ->Arg(0)->Arg(5)->Arg(10);

// ============================================================================
// BM_PutDataset / BM_GetDataset
// ============================================================================

class DatasetOpFixture : public bench::CteFixture {
public:
    void SetUp(benchmark::State& state) override {
        bench::CteFixture::SetUp(state);
        container_.emplace(CreateContainer());

        dataset_id_ = DatasetId(container_->AllocateId());
        auto shape = DatasetShape::Create({100}, {100}).value();
        DatasetMetadata meta(dataset_id_,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float64)),
            shape, *GetAllocator());
        container_->PutDataset(dataset_id_, meta);
    }
    void TearDown(benchmark::State& state) override {
        container_.reset();
        bench::CteFixture::TearDown(state);
    }
protected:
    std::optional<Container<CteBlobStore>> container_;
    DatasetId dataset_id_;
};

BENCHMARK_DEFINE_F(DatasetOpFixture, BM_PutDataset)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        ResetAllocator();
        auto shape = DatasetShape::Create({100}, {100}).value();
        DatasetMetadata meta(dataset_id_,
            DatatypeRef(PrimitiveType(PrimitiveType::Kind::Float64)),
            shape, *GetAllocator());
        state.ResumeTiming();
        container_->PutDataset(dataset_id_, meta);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(DatasetOpFixture, BM_PutDataset);

BENCHMARK_DEFINE_F(DatasetOpFixture, BM_GetDataset)(benchmark::State& state) {
    for (auto _ : state) {
        ResetAllocator();
        auto result = container_->GetDataset(dataset_id_);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(DatasetOpFixture, BM_GetDataset);

// ============================================================================
// BM_PutChunk / BM_GetChunk
// ============================================================================

class ChunkOpFixture : public bench::CteFixture {
public:
    void SetUp(benchmark::State& state) override {
        bench::CteFixture::SetUp(state);
        container_.emplace(CreateContainer());

        int64_t bytes = state.range(0);
        data_.resize(bytes);
        for (int64_t i = 0; i < bytes; ++i) {
            data_[i] = byte_t(i & 0xFF);
        }

        dataset_id_ = DatasetId(container_->AllocateId());
        uint64_t coord = 0;
        key_ = ChunkKey(dataset_id_, cstd::span<const uint64_t>(&coord, 1));
    }
    void TearDown(benchmark::State& state) override {
        container_.reset();
        bench::CteFixture::TearDown(state);
    }
protected:
    std::optional<Container<CteBlobStore>> container_;
    DatasetId dataset_id_;
    ChunkKey key_;
    std::vector<byte_t> data_;
};

BENCHMARK_DEFINE_F(ChunkOpFixture, BM_PutChunk)(benchmark::State& state) {
    cstd::span<const byte_t> span(data_.data(), data_.size());
    for (auto _ : state) {
        ResetAllocator();
        container_->PutChunk(key_, span);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0));
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(ChunkOpFixture, BM_PutChunk)
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536);

BENCHMARK_DEFINE_F(ChunkOpFixture, BM_GetChunk)(benchmark::State& state) {
    // Pre-populate chunk
    cstd::span<const byte_t> span(data_.data(), data_.size());
    container_->PutChunk(key_, span);

    std::vector<byte_t> out(data_.size());
    cstd::span<byte_t> out_span(out.data(), out.size());

    for (auto _ : state) {
        ResetAllocator();
        auto result = container_->GetChunk(key_, out_span);
        benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0));
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(ChunkOpFixture, BM_GetChunk)
    ->Arg(64)->Arg(256)->Arg(1024)->Arg(4096)->Arg(16384)->Arg(65536);

// ============================================================================
// BM_ChunkExists
// ============================================================================

class ChunkExistsFixture : public bench::CteFixture {
public:
    void SetUp(benchmark::State& state) override {
        bench::CteFixture::SetUp(state);
        container_.emplace(CreateContainer());

        dataset_id_ = DatasetId(container_->AllocateId());
        uint64_t coord_exists = 0;
        uint64_t coord_missing = 1;
        key_exists_ = ChunkKey(dataset_id_,
            cstd::span<const uint64_t>(&coord_exists, 1));
        key_missing_ = ChunkKey(dataset_id_,
            cstd::span<const uint64_t>(&coord_missing, 1));

        // Write only the "exists" key
        byte_t data[64] = {};
        container_->PutChunk(key_exists_,
            cstd::span<const byte_t>(data, sizeof(data)));
    }
    void TearDown(benchmark::State& state) override {
        container_.reset();
        bench::CteFixture::TearDown(state);
    }
protected:
    std::optional<Container<CteBlobStore>> container_;
    DatasetId dataset_id_;
    ChunkKey key_exists_;
    ChunkKey key_missing_;
};

BENCHMARK_DEFINE_F(ChunkExistsFixture, BM_ChunkExists)(benchmark::State& state) {
    bool exists = state.range(0) != 0;
    auto& key = exists ? key_exists_ : key_missing_;

    for (auto _ : state) {
        bool result = container_->ChunkExists(key);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_REGISTER_F(ChunkExistsFixture, BM_ChunkExists)
    ->Arg(1)   // exists = true
    ->Arg(0);  // exists = false
