#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../common/utility.h"
#include "../exception/exception.h"

namespace neat {
using uint = unsigned int;

namespace MNIST {
struct Image {
    Image() {}
    Image(const uint& n_pixels) : pixels(n_pixels) {}

    // Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    std::vector<double> pixels;
    std::vector<uint> label;
};

struct ImageHeader {
    uint magic_number = 0, n_rows = 0, n_columns = 0, n_images = 0;
};

struct LabelHeader {
    uint magic_number = 0, n_labels = 0;
};

ImageHeader ReadImageHeader(const std::string& file_name);
LabelHeader ReadLabelHeader(const std::string& file_name);
std::vector<Image> ReadImages(const std::string& file_name, const uint& n_images);
std::vector<uint> ReadLabels(const std::string& file_name, const uint& n_images);

std::vector<uint> Decimal2Binray(uint val);
uint Binray2Decimal(const std::vector<uint>& val);
uint Binray2Decimal(const std::vector<double>& val);
}  // namespace MNIST

}  // namespace neat
