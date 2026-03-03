#pragma once

#include "../defines.h"
#include "../serde.h"
#include <cuda/std/array>
#include <cuda/std/span>

namespace kvhdf5 {

template<size_t MaxLen>
struct gpu_string;

struct gpu_string_view {
    const char* data_ = nullptr;
    size_t length_ = 0;

    CROSS_FUN constexpr gpu_string_view() = default;

    CROSS_FUN constexpr gpu_string_view(const char* str, size_t len)
        : data_(str), length_(len) {}

    // From string literal — length known at compile time
    template<size_t N>
    CROSS_FUN constexpr gpu_string_view(const char (&str)[N])
        : data_(str), length_(N - 1) {}

    // From null-terminated C string (runtime strlen)
    CROSS_FUN gpu_string_view(const char* str) : data_(str) {
        if (str) {
            while (str[length_] != '\0') ++length_;
        }
    }

    CROSS_FUN constexpr const char* data() const { return data_; }
    CROSS_FUN constexpr size_t size() const { return length_; }
    CROSS_FUN constexpr bool empty() const { return length_ == 0; }
    CROSS_FUN constexpr char operator[](size_t i) const { return data_[i]; }

    CROSS_FUN constexpr const char* begin() const { return data_; }
    CROSS_FUN constexpr const char* end() const { return data_ + length_; }

    CROSS_FUN constexpr bool operator==(gpu_string_view other) const {
        if (length_ != other.length_) return false;
        for (size_t i = 0; i < length_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }

    CROSS_FUN constexpr bool operator!=(gpu_string_view other) const {
        return !(*this == other);
    }

    CROSS_FUN constexpr bool operator<(gpu_string_view other) const {
        size_t n = length_ < other.length_ ? length_ : other.length_;
        for (size_t i = 0; i < n; ++i) {
            if (data_[i] < other.data_[i]) return true;
            if (data_[i] > other.data_[i]) return false;
        }
        return length_ < other.length_;
    }

    CROSS_FUN constexpr bool operator<=(gpu_string_view o) const { return !(o < *this); }
    CROSS_FUN constexpr bool operator>(gpu_string_view o) const { return o < *this; }
    CROSS_FUN constexpr bool operator>=(gpu_string_view o) const { return !(*this < o); }

    CROSS_FUN constexpr gpu_string_view substr(size_t pos, size_t count = size_t(-1)) const {
        if (pos >= length_) return {};
        size_t actual = count;
        if (count == size_t(-1) || pos + count > length_)
            actual = length_ - pos;
        return {data_ + pos, actual};
    }

    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        // Write length, then characters (no null terminator)
        serde::Write(s, static_cast<uint32_t>(length_));
        s.WriteBuffer(cstd::span(reinterpret_cast<const byte_t*>(data_), length_));
    }
};

template<size_t MaxLen = 255>
struct gpu_string {
    cstd::array<char, MaxLen + 1> data_{};
    size_t length_ = 0;

    CROSS_FUN constexpr gpu_string() { data_[0] = '\0'; }

    // From string literal
    template<size_t N>
    CROSS_FUN constexpr gpu_string(const char (&str)[N]) : length_(N - 1) {
        static_assert(N <= MaxLen + 1, "String literal exceeds capacity");
        for (size_t i = 0; i < length_; ++i) data_[i] = str[i];
        data_[length_] = '\0';
    }

    // From view — explicit, truncates if too long
    CROSS_FUN explicit gpu_string(gpu_string_view v) {
        length_ = v.size() > MaxLen ? MaxLen : v.size();
        for (size_t i = 0; i < length_; ++i) data_[i] = v.data()[i];
        data_[length_] = '\0';
    }

    // Implicit conversion to view
    CROSS_FUN constexpr operator gpu_string_view() const {
        return {data_.data(), length_};
    }

    CROSS_FUN constexpr const char* data() const { return data_.data(); }
    CROSS_FUN constexpr char* data() { return data_.data(); }
    CROSS_FUN constexpr size_t size() const { return length_; }
    CROSS_FUN constexpr bool empty() const { return length_ == 0; }
    CROSS_FUN constexpr const char* c_str() const { return data_.data(); }
    CROSS_FUN constexpr static size_t max_size() { return MaxLen; }

    CROSS_FUN constexpr char operator[](size_t i) const { return data_[i]; }
    CROSS_FUN constexpr char& operator[](size_t i) { return data_[i]; }

    CROSS_FUN constexpr char* begin() { return data_.data(); }
    CROSS_FUN constexpr const char* begin() const { return data_.data(); }
    CROSS_FUN constexpr const char* cbegin() const { return data_.data(); }
    CROSS_FUN constexpr char* end() { return data_.data() + length_; }
    CROSS_FUN constexpr const char* end() const { return data_.data() + length_; }
    CROSS_FUN constexpr const char* cend() const { return data_.data() + length_; }

    // All comparisons delegate to gpu_string_view
    CROSS_FUN constexpr bool operator==(gpu_string_view o) const { return gpu_string_view(*this) == o; }
    CROSS_FUN constexpr bool operator!=(gpu_string_view o) const { return gpu_string_view(*this) != o; }
    CROSS_FUN constexpr bool operator<(gpu_string_view o) const { return gpu_string_view(*this) < o; }
    CROSS_FUN constexpr bool operator<=(gpu_string_view o) const { return gpu_string_view(*this) <= o; }
    CROSS_FUN constexpr bool operator>(gpu_string_view o) const { return gpu_string_view(*this) > o; }
    CROSS_FUN constexpr bool operator>=(gpu_string_view o) const { return gpu_string_view(*this) >= o; }

    // Returns false if appending would exceed capacity
    CROSS_FUN bool append(gpu_string_view str) {
        if (length_ + str.size() > MaxLen) return false;
        for (size_t i = 0; i < str.size(); ++i)
            data_[length_ + i] = str.data()[i];
        length_ += str.size();
        data_[length_] = '\0';
        return true;
    }

    CROSS_FUN bool push_back(char c) {
        if (length_ >= MaxLen) return false;
        data_[length_++] = c;
        data_[length_] = '\0';
        return true;
    }

    CROSS_FUN void clear() {
        length_ = 0;
        data_[0] = '\0';
    }

    // Serialization support - delegates to gpu_string_view
    template<serde::Serializer S>
    CROSS_FUN void Serialize(S& s) const {
        gpu_string_view(*this).Serialize(s);
    }

    template<serde::Deserializer D>
    CROSS_FUN static gpu_string Deserialize(D& d) {
        uint32_t len = serde::Read<uint32_t>(d);
        KVHDF5_ASSERT(len <= MaxLen, "Deserialized string exceeds capacity");

        gpu_string result;
        result.length_ = len;
        d.ReadBuffer(cstd::span(reinterpret_cast<byte_t*>(result.data_.data()), len));
        result.data_[len] = '\0';

        return result;
    }
};

} // namespace kvhdf5
