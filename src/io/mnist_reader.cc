#include "mnist_reader.h"

namespace neat {

MNIST::MNIST() {}

MNIST::ImageHeader MNIST::ReadHeader(const std::string &file_name) {
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

}  // namespace neat
