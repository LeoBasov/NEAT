#include "mnist.h"

namespace neat {
namespace MNIST {

MNIST::ImageHeader ReadImageHeader(const std::string &file_name) {
    MNIST::ImageHeader header;
    std::ifstream input(file_name, std::ios::binary);

    if (!input.is_open()) {
        throw Exception("Could not open file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    input.read((char *)&header.magic_number, sizeof(header.magic_number));
    input.read((char *)&header.n_images, sizeof(header.n_images));
    input.read((char *)&header.n_rows, sizeof(header.n_images));
    input.read((char *)&header.n_columns, sizeof(header.n_images));

    header.magic_number = utility::swap_endian<uint32_t>(header.magic_number);
    header.n_images = utility::swap_endian<uint32_t>(header.n_images);
    header.n_rows = utility::swap_endian<uint32_t>(header.n_rows);
    header.n_columns = utility::swap_endian<uint32_t>(header.n_columns);

    input.close();

    if (!input.good()) {
        throw Exception("Error occurred at reading file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    return header;
}

LabelHeader ReadLabelHeader(const std::string &file_name) {
    LabelHeader header;
    std::ifstream input(file_name, std::ios::binary);

    if (!input.is_open()) {
        throw Exception("Could not open file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    input.read((char *)&header.magic_number, sizeof(header.magic_number));
    input.read((char *)&header.n_labels, sizeof(header.n_labels));

    header.magic_number = utility::swap_endian<uint32_t>(header.magic_number);
    header.n_labels = utility::swap_endian<uint32_t>(header.n_labels);

    input.close();

    if (!input.good()) {
        throw Exception("Error occurred at reading file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    return header;
}

std::vector<MNIST::Image> ReadImages(const std::string &file_name, const uint &n_images) {
    MNIST::ImageHeader header(ReadImageHeader(file_name));
    std::vector<MNIST::Image> images(n_images, Image(header.n_columns * header.n_rows));
    std::ifstream input(file_name, std::ios::binary);
    int masgic_number(0);

    if (!input.is_open()) {
        throw Exception("Could not open file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    if (n_images > header.n_images) {
        throw Exception("Given number of images [" + std::to_string(n_images) +
                            "] > number of images given in header [" + std::to_string(header.n_images) + "]",
                        __PRETTY_FUNCTION__);
    }

    for (uint i = 0; i < 4; i++) {
        input.read((char *)&masgic_number, sizeof(masgic_number));
        masgic_number = utility::swap_endian<uint32_t>(masgic_number);
    }

    for (uint i = 0; i < n_images; i++) {
        for (uint k = 0; k < images.at(i).pixels.size(); k++) {
            unsigned char val;

            input.read((char *)&val, sizeof(val));
            images.at(i).pixels.at(k) = val;
        }
    }

    return images;
}

std::vector<uint> ReadLabels(const std::string &file_name, const uint &n_labels) {
    std::vector<uint> labels(n_labels);
    std::ifstream input(file_name, std::ios::binary);
    int number_of_items(0);

    if (!input.is_open()) {
        throw Exception("Could not open file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    input.read((char *)&number_of_items, sizeof(number_of_items));
    input.read((char *)&number_of_items, sizeof(number_of_items));

    number_of_items = utility::swap_endian<uint32_t>(number_of_items);

    if (static_cast<int>(n_labels) > number_of_items) {
        throw Exception("Given number of images [" + std::to_string(n_labels) +
                            "] > number of images given in header [" + std::to_string(number_of_items) + "]",
                        __PRETTY_FUNCTION__);
    }

    for (uint i = 0; i < n_labels; i++) {
        unsigned char val;

        input.read((char *)&val, sizeof(val));
        labels.at(i) = val;
    }

    input.close();

    if (!input.good()) {
        throw Exception("Error occurred at reading file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    return labels;
}

std::vector<uint> Decimal2Binray(uint val) {
    std::vector<uint> retval;

    while (val) {
        retval.push_back(val % 2);
        val /= 2;
    }

    while (retval.size() < 8) {
        retval.push_back(0);
    }

    return std::vector<uint>(retval.rbegin(), retval.rend());
}

uint Binray2Decimal(const std::vector<uint> &val) {
    uint retval(0);

    for (uint i = 0; i < val.size(); i++) {
        if (val.at(i)) {
            retval += std::pow(2, val.size() - i - 1);
        }
    }

    return retval;
}

uint Binray2Decimal(const std::vector<double> &val) {
    uint retval(0);

    for (uint i = 0; i < val.size(); i++) {
        if (std::round(val.at(i))) {
            retval += std::pow(2, val.size() - i - 1);
        }
    }

    return retval;
}

}  // namespace MNIST
}  // namespace neat
