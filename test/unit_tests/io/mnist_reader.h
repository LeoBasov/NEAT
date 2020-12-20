#pragma once

#include <gtest/gtest.h>

#include "../../../src/io/mnist_reader.h"

namespace neat {

TEST(MNIST, ConverToBinray) {
    const uint val1(255), val2(15), val3(0);
    const std::vector<uint> ref1(8, 1), ref2({0, 0, 0, 0, 1, 1, 1, 1}), ref3(8, 0);
    std::vector<uint> result1, result2, result3;

    result1 = MNIST::ConverToBinray(val1);
    result2 = MNIST::ConverToBinray(val2);
    result3 = MNIST::ConverToBinray(val3);

    ASSERT_EQ(8, result1.size());
    ASSERT_EQ(8, result2.size());
    ASSERT_EQ(8, result3.size());

    for (uint i = 0; i < 8; i++) {
        ASSERT_EQ(ref1.at(i), result1.at(i));
        ASSERT_EQ(ref2.at(i), result2.at(i));
        ASSERT_EQ(ref3.at(i), result3.at(i));
    }
}

}  // namespace neat
