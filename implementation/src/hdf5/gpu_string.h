#pragma once

#include "error.h"

namespace hdf5 {

// Forward declaration
template<size_t MaxLen>
struct gpu_string;

/**
 * gpu_string_view - Non-owning reference to a string
 * Similar to std::string_view but CUDA-compatible
 * Can point to literals, gpu_string, or raw char arrays
 */
struct gpu_string_view {
    const char* data_;
    size_t length_;

    // Construct from pointer and length
    constexpr gpu_string_view(const char* str, size_t len)
        : data_(str), length_(len) {}

    // Construct from null-terminated string literal
    template<size_t N>
    constexpr gpu_string_view(const char (&str)[N])
        : data_(str), length_(N - 1) {}  // -1 for null terminator

    // Default constructor - empty view
    constexpr gpu_string_view()
        : data_(nullptr), length_(0) {}

    // Accessors
    constexpr const char* data() const { return data_; }
    constexpr size_t size() const { return length_; }
    constexpr size_t length() const { return length_; }
    constexpr bool empty() const { return length_ == 0; }

    // Indexing
    constexpr char operator[](size_t idx) const {
        return data_[idx];
    }

    // Comparison
    constexpr bool operator==(gpu_string_view other) const {
        if (length_ != other.length_) return false;
        for (size_t i = 0; i < length_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }

    constexpr bool operator!=(gpu_string_view other) const {
        return !(*this == other);
    }

    // Lexicographic comparison
    constexpr bool operator<(gpu_string_view other) const {
        size_t min_len = length_ < other.length_ ? length_ : other.length_;
        for (size_t i = 0; i < min_len; ++i) {
            if (data_[i] < other.data_[i]) return true;
            if (data_[i] > other.data_[i]) return false;
        }
        return length_ < other.length_;
    }

    constexpr bool operator<=(gpu_string_view other) const {
        return other >= *this;
    }

    constexpr bool operator>(gpu_string_view other) const {
        return other < *this;
    }

    constexpr bool operator>=(gpu_string_view other) const {
        return !(*this < other);
    }

    // Substring
    constexpr gpu_string_view substr(size_t pos, size_t count = size_t(-1)) const {
        if (pos > length_) {
            return gpu_string_view(data_, 0);
        }
        size_t actual_count = count;
        if (pos + count > length_ || count == size_t(-1)) {
            actual_count = length_ - pos;
        }
        return gpu_string_view(data_ + pos, actual_count);
    }
};

/**
 * gpu_string - Owning fixed-size string
 * Stack-allocated, CUDA-compatible string with maximum length
 * Always null-terminated for C API compatibility and debugging
 * Default MaxLen = 255 so array size is power-of-2 (256 bytes) for better alignment
 */
template<size_t MaxLen = 255>
struct gpu_string {
    // Internal storage: MaxLen + 1 for null terminator
    cstd::array<char, MaxLen + 1> data_;
    size_t length_ = 0;

    // Default constructor - empty string
    constexpr gpu_string() {
        data_[0] = '\0';
    }

    // Construct from null-terminated string literal
    template<size_t N>
    constexpr gpu_string(const char (&str)[N]) : length_(N - 1) {
        static_assert(N <= MaxLen + 1, "String literal exceeds maximum length");
        for (size_t i = 0; i < length_; ++i) {
            data_[i] = str[i];
        }
        data_[length_] = '\0';
    }

    // Construct from view (explicit to prevent accidental copies)
    explicit gpu_string(gpu_string_view view) {
        if (view.size() > MaxLen) {
            // Truncate to MaxLen
            length_ = MaxLen;
        } else {
            length_ = view.size();
        }

        for (size_t i = 0; i < length_; ++i) {
            data_[i] = view.data()[i];
        }
        data_[length_] = '\0';
    }

    // Construct from pointer and length with error handling
    static hdf5::expected<gpu_string> from_chars(const char* str, size_t len) {
        if (len > MaxLen) {
            return hdf5::error(
                hdf5::HDF5ErrorCode::BufferTooSmall,
                "String length exceeds maximum"
            );
        }

        gpu_string result;
        result.length_ = len;
        for (size_t i = 0; i < len; ++i) {
            result.data_[i] = str[i];
        }
        result.data_[len] = '\0';

        return result;
    }

    // Implicit conversion TO view (allows passing gpu_string where view is expected)
    constexpr operator gpu_string_view() const {
        return gpu_string_view(data_.data(), length_);
    }

    // Accessors
    constexpr const char* data() const { return data_.data(); }
    constexpr char* data() { return data_.data(); }
    constexpr size_t size() const { return length_; }
    constexpr size_t length() const { return length_; }
    constexpr bool empty() const { return length_ == 0; }
    constexpr const char* c_str() const { return data_.data(); }
    constexpr static size_t max_size() { return MaxLen; }

    // Indexing
    constexpr char operator[](size_t idx) const {
        return data_[idx];
    }

    constexpr char& operator[](size_t idx) {
        return data_[idx];
    }

    // Comparison (delegate to view)
    constexpr bool operator==(gpu_string_view other) const {
        return gpu_string_view(*this) == other;
    }

    constexpr bool operator!=(gpu_string_view other) const {
        return gpu_string_view(*this) != other;
    }

    constexpr bool operator<(gpu_string_view other) const {
        return gpu_string_view(*this) < other;
    }

    constexpr bool operator<=(gpu_string_view other) const {
        return gpu_string_view(*this) <= other;
    }

    constexpr bool operator>(gpu_string_view other) const {
        return gpu_string_view(*this) > other;
    }

    constexpr bool operator>=(gpu_string_view other) const {
        return gpu_string_view(*this) >= other;
    }

    // String modification operations

    /**
     * Append a string view to this string
     * Returns error if result would exceed MaxLen
     */
    hdf5::expected<void> append(gpu_string_view str) {
        if (length_ + str.size() > MaxLen) {
            return hdf5::error(
                hdf5::HDF5ErrorCode::BufferTooSmall,
                "String append would exceed maximum length"
            );
        }

        for (size_t i = 0; i < str.size(); ++i) {
            data_[length_ + i] = str.data()[i];
        }
        length_ += str.size();
        data_[length_] = '\0';

        return {};
    }

    /**
     * Append a single character
     * Returns error if result would exceed MaxLen
     */
    hdf5::expected<void> push_back(char c) {
        if (length_ >= MaxLen) {
            return hdf5::error(
                hdf5::HDF5ErrorCode::BufferTooSmall,
                "Cannot append character, string at maximum length"
            );
        }

        data_[length_] = c;
        ++length_;
        data_[length_] = '\0';

        return {};
    }

    /**
     * Clear the string
     */
    void clear() {
        length_ = 0;
        data_[0] = '\0';
    }

    /**
     * Resize the string
     * If new_size > current size, fills with null bytes
     * Returns error if new_size > MaxLen
     */
    hdf5::expected<void> resize(size_t new_size, char fill = '\0') {
        if (new_size > MaxLen) {
            return hdf5::error(
                hdf5::HDF5ErrorCode::BufferTooSmall,
                "Resize would exceed maximum length"
            );
        }

        if (new_size > length_) {
            for (size_t i = length_; i < new_size; ++i) {
                data_[i] = fill;
            }
        }

        length_ = new_size;
        data_[length_] = '\0';

        return {};
    }

    /**
     * Get a substring as a view
     */
    constexpr gpu_string_view substr(size_t pos, size_t count = size_t(-1)) const {
        return gpu_string_view(*this).substr(pos, count);
    }
};

} // namespace hdf5
