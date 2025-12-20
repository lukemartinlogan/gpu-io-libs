#pragma once

#include "cstd.h"

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

    __device__
    constexpr cstd::unexpected<HDF5Error> error(const HDF5ErrorCode code, const char* desc = nullptr) {
        return cstd::unexpected(HDF5Error{ .code = code, .description = desc });
    }
}