#include <fstream>
#include <iostream>

#include "../../src/io/mnist_reader.h"

using namespace neat;

int main(int, char**) {
    // Data set found here: http://yann.lecun.com/exdb/mnist/

    MNIST::ImageHeader image_header;
    MNIST reader;

    image_header = reader.ReadHeader("/home/lbasov/AI/train-images-idx3-ubyte");

    std::cout << "N IMAGES READ: " << image_header.n_images << " EXPECTED: 60000" << std::endl;
    std::cout << "N ROWS:        " << image_header.n_rows << " EXPECTED: 28" << std::endl;
    std::cout << "N COLUMNS:     " << image_header.n_columns << " EXPECTED: 28" << std::endl;

    return 0;
}
