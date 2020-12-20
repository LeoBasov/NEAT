#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../common/utility.h"
#include "../exception/exception.h"

namespace neat {
using uint = unsigned int;
class MNIST {
   public:
    struct Image {
        // Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
        std::vector<uint> pixes;
    };

    struct ImageHeader {
        uint n_rows = 0, n_columns = 0, n_images = 0;
    };

    MNIST();
    ~MNIST() = default;

    ImageHeader ReadHeader(const std::string& file_name);
};
}  // namespace neat
