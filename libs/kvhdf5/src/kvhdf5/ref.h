#pragma once
#include "defines.h"

namespace kvhdf5 {

template<typename T>
class Ref {
    T* ptr_;
public:
    CROSS_FUN explicit Ref(T& ref) : ptr_(&ref) {}
    CROSS_FUN T* operator->() const { return ptr_; }
    CROSS_FUN T& operator*() const { return *ptr_; }
    CROSS_FUN T* get() const { return ptr_; }
};

} // namespace kvhdf5
