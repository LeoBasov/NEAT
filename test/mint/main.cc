#include <fstream>
#include <iostream>

#include "../../src/io/mnist_reader.h"

using namespace neat;

int main(int, char**) {
    // Data set found here: http://yann.lecun.com/exdb/mnist/

    MNIST::ImageHeader image_header;
    MNIST reader;
    std::vector<MNIST::Image> images;
    std::vector<uint> labels;
    const uint n_images(5000);
    const std::string file_name_images("/home/lbasov/AI/train-images-idx3-ubyte");
    const std::string file_name_labels("/home/lbasov/AI/train-labels-idx1-ubyte");

    image_header = reader.ReadImageHeader(file_name_images);

    std::cout << "N IMAGES READ: " << image_header.n_images << " EXPECTED: 60000" << std::endl;
    std::cout << "N ROWS:        " << image_header.n_rows << " EXPECTED: 28" << std::endl;
    std::cout << "N COLUMNS:     " << image_header.n_columns << " EXPECTED: 28" << std::endl;

    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " IMAGES" << std::endl;

    images = reader.ReadImages(file_name_images, n_images);

    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " LABELS" << std::endl;

    labels = reader.ReadLabels(file_name_labels, n_images);

    std::cout << "========================================================================" << std::endl;

    return 0;
}
