#pragma once

#include <gtest/gtest.h>

#include "../../../src/io/mnist.h"

namespace neat {

TEST(MNIST, Decimal2Binray) {
    const uint val1(255), val2(15), val3(0);
    const std::vector<uint> ref1(8, 1), ref2({0, 0, 0, 0, 1, 1, 1, 1}), ref3(8, 0);
    std::vector<uint> result1, result2, result3;

    result1 = MNIST::Decimal2Binray(val1);
    result2 = MNIST::Decimal2Binray(val2);
    result3 = MNIST::Decimal2Binray(val3);

    ASSERT_EQ(8, result1.size());
    ASSERT_EQ(8, result2.size());
    ASSERT_EQ(8, result3.size());

    for (uint i = 0; i < 8; i++) {
        ASSERT_EQ(ref1.at(i), result1.at(i));
        ASSERT_EQ(ref2.at(i), result2.at(i));
        ASSERT_EQ(ref3.at(i), result3.at(i));
    }
}

TEST(MNIST, Binray2Decimal) {
    const uint val1(255), val2(15), val3(0);
    const std::vector<uint> ref1(8, 1), ref2({0, 0, 0, 0, 1, 1, 1, 1}), ref3(8, 0);
    uint result1, result2, result3;

    result1 = MNIST::Binray2Decimal(ref1);
    result2 = MNIST::Binray2Decimal(ref2);
    result3 = MNIST::Binray2Decimal(ref3);

    ASSERT_EQ(val1, result1);
    ASSERT_EQ(val2, result2);
    ASSERT_EQ(val3, result3);
}

}  // namespace neat
