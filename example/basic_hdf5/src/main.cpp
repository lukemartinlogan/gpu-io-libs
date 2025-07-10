#include <H5Cpp.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include "hwinfo.h"
#include "sample.h"

int main() {
    std::filesystem::path file_path = "./system_info.h5";

    std::string group_name;
    uint64_t sample_ms;
    uint16_t sample_ct;

    std::cout << "Group name: ";
    std::getline(std::cin, group_name);

    std::cout << "Sample count: ";
    if (!(std::cin >> sample_ct)){
        throw std::runtime_error("given sample count was not integer");
    }

    std::cout << "Sample interval (ms): ";
    if (!(std::cin >> sample_ms)){
        throw std::runtime_error("given sample interval was not integer");
    }

    try {
        unsigned int flags = std::filesystem::exists(file_path) ? H5F_ACC_RDWR : H5F_ACC_TRUNC;

        H5::H5File file(file_path, flags);

        H5::Group run_group = file.createGroup(group_name);

        hsize_t dims[1] = { sample_ct };
        H5::DataSpace dataspace(1, dims);

        H5::CompType data_type = Sample::h5_comp_ty();

        H5::DataSet dataset = run_group.createDataSet("samples", data_type, dataspace);

        std::vector<Sample> samples;
        samples.reserve(sample_ct);

        for (size_t s = 0; s < sample_ct; ++s) {
            samples.push_back({
                .timestamp_ms = current_time_ms(),
                .memory_kb = mem_kb(),
                .cpu_time = cpu_time(),
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms));
        }

        dataset.write(samples.data(), data_type);

        H5::StrType str_ty(H5::PredType::C_S1, H5T_VARIABLE);
        H5::DataSpace scalar_space(H5S_SCALAR);

        H5::Attribute unit_attr = run_group.createAttribute("units", str_ty, scalar_space);
        unit_attr.write(str_ty, "timestamp: ms, cpu: %, memory: kB");

        run_group.createAttribute("sampling_interval_ms", H5::PredType::NATIVE_UINT64, scalar_space)
            .write(H5::PredType::NATIVE_UINT64, &sample_ms);

        run_group.createAttribute("sample_count", H5::PredType::NATIVE_UINT16, scalar_space)
            .write(H5::PredType::NATIVE_UINT16, &sample_ct);

        std::cout << std::format("wrote data to {} in {}.", dataset.getObjName(), file_path.string()) << std::endl;

        return 0;
    } catch (H5::Exception& e) {
        std::cout << "HDF5 error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "unknown error" << std::endl;
    }
}
