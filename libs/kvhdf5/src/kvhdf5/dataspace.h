#pragma once

#include "defines.h"
#include "error.h"
#include <cuda/std/array>
#include <cuda/std/span>

namespace kvhdf5 {

enum class SelectionOp : uint8_t {
    Set,
    Or,
    And,
    Xor,
    NotA,
    NotB,
};

enum class SelectionType : uint8_t {
    All,
    None,
    Hyperslab,
};

struct HyperslabSelection {
    cstd::array<uint64_t, MAX_DIMS> start{};
    cstd::array<uint64_t, MAX_DIMS> stride{};
    cstd::array<uint64_t, MAX_DIMS> count{};
    cstd::array<uint64_t, MAX_DIMS> block{};
    uint8_t ndims{0};

    CROSS_FUN cstd::span<const uint64_t> Start() const {
        return cstd::span(start.data(), ndims);
    }
    CROSS_FUN cstd::span<uint64_t> Start() {
        return cstd::span(start.data(), ndims);
    }

    CROSS_FUN cstd::span<const uint64_t> Stride() const {
        return cstd::span(stride.data(), ndims);
    }
    CROSS_FUN cstd::span<uint64_t> Stride() {
        return cstd::span(stride.data(), ndims);
    }

    CROSS_FUN cstd::span<const uint64_t> Count() const {
        return cstd::span(count.data(), ndims);
    }
    CROSS_FUN cstd::span<uint64_t> Count() {
        return cstd::span(count.data(), ndims);
    }

    CROSS_FUN cstd::span<const uint64_t> Block() const {
        return cstd::span(block.data(), ndims);
    }
    CROSS_FUN cstd::span<uint64_t> Block() {
        return cstd::span(block.data(), ndims);
    }

    CROSS_FUN constexpr bool operator==(const HyperslabSelection&) const = default;
};

class Dataspace {
    cstd::array<uint64_t, MAX_DIMS> dims_{};
    cstd::array<uint64_t, MAX_DIMS> max_dims_{};
    uint8_t ndims_{0};
    SelectionType sel_type_{SelectionType::All};
    HyperslabSelection hyperslab_{};

    CROSS_FUN Dataspace() = default;

public:
    CROSS_FUN static expected<Dataspace> CreateSimple(
        cstd::span<const uint64_t> dims,
        cstd::span<const uint64_t> max_dims = {}
    ) {
        if (dims.size() == 0 || dims.size() > MAX_DIMS) {
            return make_error(ErrorCode::InvalidArgument,
                "ndims must be between 1 and MAX_DIMS");
        }
        if (!max_dims.empty() && max_dims.size() != dims.size()) {
            return make_error(ErrorCode::InvalidArgument,
                "max_dims must have same rank as dims");
        }

        Dataspace ds;
        ds.ndims_ = static_cast<uint8_t>(dims.size());

        for (uint8_t i = 0; i < ds.ndims_; ++i) {
            ds.dims_[i] = dims[i];
            ds.max_dims_[i] = max_dims.empty() ? dims[i] : max_dims[i];
        }

        return ds;
    }

    CROSS_FUN static Dataspace CreateScalar() {
        return Dataspace{};
    }

    CROSS_FUN uint8_t GetNDims() const { return ndims_; }

    CROSS_FUN void GetDims(
        cstd::span<uint64_t> out_dims,
        cstd::span<uint64_t> out_max_dims = {}
    ) const {
        KVHDF5_ASSERT(out_dims.size() >= ndims_, "output dims span too small");

        for (uint8_t i = 0; i < ndims_; ++i) {
            out_dims[i] = dims_[i];
        }
        if (!out_max_dims.empty()) {
            KVHDF5_ASSERT(out_max_dims.size() >= ndims_,
                "output max_dims span too small");
            for (uint8_t i = 0; i < ndims_; ++i) {
                out_max_dims[i] = max_dims_[i];
            }
        }
    }

    CROSS_FUN uint64_t GetTotalElements() const {
        if (ndims_ == 0) return 1;
        uint64_t total = 1;
        for (uint8_t i = 0; i < ndims_; ++i) {
            total *= dims_[i];
        }
        return total;
    }

    CROSS_FUN uint64_t GetSelectedPointCount() const {
        switch (sel_type_) {
            case SelectionType::All:
                return GetTotalElements();
            case SelectionType::None:
                return 0;
            case SelectionType::Hyperslab: {
                uint64_t total = 1;
                for (uint8_t i = 0; i < hyperslab_.ndims; ++i) {
                    total *= hyperslab_.count[i] * hyperslab_.block[i];
                }
                return total;
            }
        }
        KVHDF5_ASSERT(false, "GetSelectedPointCount: unreachable");
        return 0;
    }

    CROSS_FUN SelectionType GetSelectionType() const { return sel_type_; }

    CROSS_FUN const HyperslabSelection& GetHyperslab() const { return hyperslab_; }

    CROSS_FUN cstd::span<const uint64_t> GetDimsSpan() const {
        return cstd::span<const uint64_t>(dims_.data(), ndims_);
    }

    CROSS_FUN cstd::span<const uint64_t> GetMaxDimsSpan() const {
        return cstd::span<const uint64_t>(max_dims_.data(), ndims_);
    }

    CROSS_FUN void SelectAll() { sel_type_ = SelectionType::All; }

    CROSS_FUN void SelectNone() { sel_type_ = SelectionType::None; }

    CROSS_FUN expected<void> SelectHyperslab(
        SelectionOp op,
        cstd::span<const uint64_t> start,
        cstd::span<const uint64_t> stride,
        cstd::span<const uint64_t> count,
        cstd::span<const uint64_t> block
    ) {
        (void)op;

        if (start.size() != ndims_ || count.size() != ndims_) {
            return make_error(ErrorCode::InvalidArgument,
                "start/count rank must match dataspace rank");
        }
        if (!stride.empty() && stride.size() != ndims_) {
            return make_error(ErrorCode::InvalidArgument,
                "stride rank must match dataspace rank");
        }
        if (!block.empty() && block.size() != ndims_) {
            return make_error(ErrorCode::InvalidArgument,
                "block rank must match dataspace rank");
        }

        HyperslabSelection sel;
        sel.ndims = ndims_;

        for (uint8_t i = 0; i < ndims_; ++i) {
            sel.start[i] = start[i];
            sel.stride[i] = stride.empty() ? 1 : stride[i];
            sel.count[i] = count[i];
            sel.block[i] = block.empty() ? 1 : block[i];

            if (sel.count[i] > 0 && sel.block[i] > 0) {
                uint64_t last = sel.start[i]
                    + (sel.count[i] - 1) * sel.stride[i]
                    + (sel.block[i] - 1);
                if (last >= dims_[i]) {
                    return make_error(ErrorCode::InvalidArgument,
                        "hyperslab selection exceeds dataspace bounds");
                }
            }
        }

        hyperslab_ = sel;
        sel_type_ = SelectionType::Hyperslab;
        return {};
    }

    CROSS_FUN void SetDims(cstd::span<const uint64_t> new_dims) {
        KVHDF5_ASSERT(new_dims.size() == ndims_, "SetDims: rank mismatch");
        for (uint8_t i = 0; i < ndims_; ++i) {
            dims_[i] = new_dims[i];
        }
    }

    CROSS_FUN constexpr bool operator==(const Dataspace&) const = default;
};

} // namespace kvhdf5
