#pragma once
#include <chrono>
#include <fstream>

inline uint64_t current_time_ms() {
    using namespace std::chrono;

    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

inline uint64_t mem_kb() {
    std::ifstream meminfo("/proc/meminfo");

    std::string key;
    uint64_t val;
    std::string unit;

    while (meminfo >> key >> val >> unit){
        if (key.contains("MemTotal")) {
            return val;
        }
    }

    throw std::runtime_error("can't retrieve MemTotal from /proc/meminfo");
}

inline uint64_t cpu_time() {
    std::ifstream stat("/proc/stat");

    std::string ignore;
    uint64_t user, nice, system, idle, iowait, irq, softirq;

    stat >> ignore >> user >> nice >> system >> idle >> iowait >> irq >> softirq;

    return user + nice + system + idle + iowait + irq + softirq;
}
