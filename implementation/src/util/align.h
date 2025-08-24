#pragma once

template<typename T>
T EightBytesAlignedSize(T val) {
    return val + 7 & ~7;
}
