#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

using byte_t = std::byte;

using offset_t = uint64_t;
using len_t = uint64_t;

constexpr offset_t kUndefinedOffset = std::numeric_limits<offset_t>::max();

namespace hdf5 {
    constexpr uint32_t MAX_DIMS = 8;
}

#ifdef LIBCUDACXX_AVAILABLE
    #include <cuda/std/optional>
    #include <cuda/std/array>
    #include <cuda/std/variant>
    #include <cuda/std/span>
    #include <cuda/std/utility>
    #include <cuda/std/bitset>
    #include <cuda/std/tuple>
    #include <cuda/std/chrono>
    #include <cuda/std/inplace_vector>
    #include <cuda/std/expected>

    namespace cstd = cuda::std;

    namespace hdf5 {
        template<typename T>
        using dim_vector = cstd::inplace_vector<T, MAX_DIMS>;
    }
#endif

namespace hdf5 {
    enum class HDF5ErrorCode {
        // File I/O Errors
        FileOpenFailed,
        FileSeekFailed,
        FilePositionFailed,

        // Format/Validation Errors
        InvalidSignature,
        InvalidVersion,
        InvalidChecksum,
        InvalidType,
        InvalidClass,
        InvalidFlags,
        InvalidTerminator,
        SizeMismatch,

        // Bounds/Range Errors
        IndexOutOfBounds,
        CoordinateOutOfBounds,
        SelectionOutOfBounds,
        SelectionOverflow,
        IteratorAtEnd,

        // Invalid Arguments
        BufferTooSmall,
        BufferTooLarge,
        EmptyParameter,
        ZeroParameter,
        DimensionMismatch,
        BlockOverlap,
        InvalidParameterCombination,

        // Not Implemented
        NotImplemented,
        FeatureNotSupported,
        DeprecatedFeature,

        // Logic Errors (Internal consistency/state errors)
        InvalidVariantState,
        WrongNodeType,
        EmptyNode,
        CapacityExceeded,
        AllocationMismatch,
        FieldPresenceMismatch,

        // Data Errors
        StringNotNullTerminated,
        IncorrectByteCount,
        InvalidDataValue,

        // Resource/Limit Errors
        BTreeOverflow,
        MaxDimensionsExceeded,
        BufferNotAligned
    };

    struct HDF5Error {
        HDF5ErrorCode code;
        const char* description;
    };

    template<typename T>
    using expected = cstd::expected<T, HDF5Error>;

    constexpr cstd::unexpected<HDF5Error> error(const HDF5ErrorCode code, const char* desc = nullptr) {
        return cstd::unexpected(HDF5Error{ .code = code, .description = desc });
    }
}