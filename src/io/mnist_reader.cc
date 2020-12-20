#include "mnist_reader.h"

namespace neat {

MNIST::MNIST() {}

MNIST::ImageHeader MNIST::ReadHeader(const std::string &file_name) const {
    MNIST::ImageHeader header;
    std::ifstream input(file_name, std::ios::binary);
    int masgic_number(0);

    if (!input.is_open()) {
        throw Exception("Could not open file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    input.read((char *)&masgic_number, sizeof(masgic_number));

    input.read((char *)&header.n_images, sizeof(header.n_images));
    input.read((char *)&header.n_rows, sizeof(header.n_images));
    input.read((char *)&header.n_columns, sizeof(header.n_images));

    header.n_images = utility::swap_endian<uint32_t>(header.n_images);
    header.n_rows = utility::swap_endian<uint32_t>(header.n_rows);
    header.n_columns = utility::swap_endian<uint32_t>(header.n_columns);

    input.close();

    if (!input.good()) {
        throw Exception("Error occurred at reading file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    return header;
}

std::vector<MNIST::Image> MNIST::ReadImages(const std::string &file_name, const uint &n_images) const {
    MNIST::ImageHeader header(ReadHeader(file_name));
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
        std::cout << masgic_number << std::endl;
    }

    for (uint i = 0; i < n_images; i++) {
        for (uint k = 0; k < images.at(i).pixes.size(); k++) {
            unsigned char val;

            input.read((char *)&val, sizeof(val));
            images.at(i).pixes.at(k) = val;

            std::cout << images.at(i).pixes.at(k) << std::endl;
        }
    }

    return images;
}

}  // namespace neat
