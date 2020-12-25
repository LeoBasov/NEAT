#pragma once

#include <gtest/gtest.h>

#include "../../../src/io/mnist.h"

namespace neat {

const std::string file_name_images("./test/unit_tests/test_data/t10k-images-idx3-ubyte");
const std::string file_name_labels("./test/unit_tests/test_data/t10k-labels-idx1-ubyte");

TEST(mnist, ReadImageHeader) {
    mnist::ImageHeader header(mnist::ReadImageHeader(file_name_images));

    ASSERT_THROW(mnist::ReadImageHeader(""), Exception);

    ASSERT_EQ(2051, header.magic_number);
    ASSERT_EQ(10000, header.n_images);
    ASSERT_EQ(28, header.n_rows);
    ASSERT_EQ(28, header.n_columns);
}

TEST(mnist, ReadLabelHeader) {
    mnist::LabelHeader header(mnist::ReadLabelHeader(file_name_labels));

    ASSERT_THROW(mnist::ReadLabelHeader(""), Exception);

    ASSERT_EQ(2049, header.magic_number);
    ASSERT_EQ(10000, header.n_labels);
}

TEST(mnist, ReadLabels) {
    const uint n_lables(5);
    std::vector<uint> labels(mnist::ReadLabels(file_name_labels, n_lables));

    ASSERT_THROW(mnist::ReadLabels("", n_lables), Exception);
    ASSERT_THROW(mnist::ReadLabels(file_name_labels, 10001), Exception);

    ASSERT_EQ(n_lables, labels.size());

    ASSERT_EQ(7, labels.at(0));
    ASSERT_EQ(2, labels.at(1));
    ASSERT_EQ(1, labels.at(2));
    ASSERT_EQ(0, labels.at(3));
    ASSERT_EQ(4, labels.at(4));
}

TEST(mnist, ReadImages) {
    const uint n_images(5);
    const uint n_pixels(28 * 28);
    std::vector<mnist::Image> images(mnist::ReadImages(file_name_images, n_images));

    ASSERT_THROW(mnist::ReadImages("", n_images), Exception);
    ASSERT_THROW(mnist::ReadImages(file_name_images, 10001), Exception);

    ASSERT_EQ(n_images, images.size());

    for (auto image : images) {
        ASSERT_EQ(n_pixels, image.pixels.size());

        for (uint i = 0; i < image.pixels.size(); i++) {
            ASSERT_TRUE(image.pixels.at(i) < 256);

            if (i < 28 * 3) {
                ASSERT_TRUE(image.pixels.at(i) == 0);
            }
        }
    }
}

TEST(mnist, Decimal2Binray) {
    const uint val1(255), val2(15), val3(0);
    const std::vector<uint> ref1(8, 1), ref2({0, 0, 0, 0, 1, 1, 1, 1}), ref3(8, 0);
    std::vector<uint> result1, result2, result3;

    result1 = mnist::Decimal2Binray(val1);
    result2 = mnist::Decimal2Binray(val2);
    result3 = mnist::Decimal2Binray(val3);

    ASSERT_EQ(8, result1.size());
    ASSERT_EQ(8, result2.size());
    ASSERT_EQ(8, result3.size());

    for (uint i = 0; i < 8; i++) {
        ASSERT_EQ(ref1.at(i), result1.at(i));
        ASSERT_EQ(ref2.at(i), result2.at(i));
        ASSERT_EQ(ref3.at(i), result3.at(i));
    }
}

TEST(mnist, Binray2Decimal) {
    const uint val1(255), val2(15), val3(0);
    const std::vector<uint> ref1(8, 1), ref2({0, 0, 0, 0, 1, 1, 1, 1}), ref3(8, 0);
    const std::vector<double> ref4(8, 1.0), ref5({0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0}), ref6(8, 0.0);
    uint result1, result2, result3, result4, result5, result6;

    result1 = mnist::Binray2Decimal(ref1);
    result2 = mnist::Binray2Decimal(ref2);
    result3 = mnist::Binray2Decimal(ref3);

    result4 = mnist::Binray2Decimal(ref4);
    result5 = mnist::Binray2Decimal(ref5);
    result6 = mnist::Binray2Decimal(ref6);

    ASSERT_EQ(val1, result1);
    ASSERT_EQ(val2, result2);
    ASSERT_EQ(val3, result3);

    ASSERT_EQ(val1, result4);
    ASSERT_EQ(val2, result5);
    ASSERT_EQ(val3, result6);
}

}  // namespace neat
