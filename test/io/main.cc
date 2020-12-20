#include <fstream>
#include <iostream>

#include "../../src/io/mnist_reader.h"

using namespace neat;

int main(int, char**) {
    // Data set found here: http://yann.lecun.com/exdb/mnist/

    MNIST::ImageHeader image_header;
    MNIST reader;
    std::vector<MNIST::Image> images;
    const uint n_images(5);
    const std::string file_name("/home/lbasov/AI/train-images-idx3-ubyte");

    image_header = reader.ReadHeader(file_name);

    std::cout << "N IMAGES READ: " << image_header.n_images << " EXPECTED: 60000" << std::endl;
    std::cout << "N ROWS:        " << image_header.n_rows << " EXPECTED: 28" << std::endl;
    std::cout << "N COLUMNS:     " << image_header.n_columns << " EXPECTED: 28" << std::endl;

    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " IMAGES" << std::endl;

    images = reader.ReadImages(file_name, n_images);

    for (uint i = 0; i < images.front().pixes.size(); i++) {
        if (images.front().pixes.at(i) > 255) {
            throw Exception(std::to_string(images.front().pixes.at(i)));
        }
    }

    for (uint p = 0; p < n_images; p++) {
        for (uint i = 0, j = 0; i < image_header.n_rows * image_header.n_columns; i++, j++) {
            if (j == 28) {
                std::cout << std::endl;
                j = 0;
            }

            if (images.at(p).pixes.at(i)) {
                std::cout << " ";
            } else {
                std::cout << "█";
            }
        }

        std::cout << std::endl;
    }

    std::cout << "========================================================================" << std::endl;

    return 0;
}
