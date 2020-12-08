#pragma once

#include <gtest/gtest.h>

#include "../../../src/common/utility.h"

namespace neat {
namespace utility {

TEST(utility, Sigmoid) {
    const double paramter(4.9);
    const double val1(-1000.0), val2(0.0), val3(1000.0);
    const double result1(0.0), result2(0.5), result3(1.0);

    ASSERT_DOUBLE_EQ(result1, Sigmoid(val1, paramter));
    ASSERT_DOUBLE_EQ(result2, Sigmoid(val2, paramter));
    ASSERT_DOUBLE_EQ(result3, Sigmoid(val3, paramter));
}

}  // namespace utility
}  // namespace neat
