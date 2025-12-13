#include "hdf5/file.h"
#include <iostream>

int main() {
    return 0;

    // try {
    //     std::cout << std::filesystem::current_path() << std::endl;
    //
    //     auto file = File::New("../../data/system_info.h5");
    //
    //     if (!file) {
    //         throw std::runtime_error(file.error().description);
    //     }
    //
    //     std::cout << file->RootGroup().GetLocalHeap().BufferSize() << std::endl;
    // }
    // catch (const std::exception& e) {
    //     std::cout << e.what() << std::endl;
    // }
}
