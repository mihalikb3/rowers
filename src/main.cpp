#include "Simulation.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

using namespace modern;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " [-C|-A|-R] [-O|-R] input_file1 [input_file2 ...]" << std::endl;
        std::cerr << "  -C|-A|-R: Geometry (Chain, Arbitrary, Ring)" << std::endl;
        std::cerr << "  -O|-R: Tensor (Oseen, Rotne-Prager)" << std::endl;
        return -1;
    }

    GeometryType geometry = GeometryType::RING;
    if (std::strncmp(argv[1], "-C", 2) == 0) {
        geometry = GeometryType::CHAIN;
    } else if (std::strncmp(argv[1], "-A", 2) == 0) {
        geometry = GeometryType::ARBITRARY;
    } else if (std::strncmp(argv[1], "-R", 2) == 0) {
        geometry = GeometryType::RING;
    }

    bool use_rotne_prager = true;
    if (std::strncmp(argv[2], "-O", 2) == 0) {
        use_rotne_prager = false;
    } else {
        use_rotne_prager = true;
    }

    for (int i = 3; i < argc; ++i) {
        try {
            std::string input_file = argv[i];
            Simulation sim(input_file, geometry, use_rotne_prager);
            sim.run();
        } catch (const std::exception& e) {
            std::cerr << "Error running simulation for " << argv[i] << ": " << e.what() << std::endl;
        }
    }

    return 0;
}
