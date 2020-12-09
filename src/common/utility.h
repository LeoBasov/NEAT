#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace neat {
namespace utility {

double Sigmoid(const double& value, const double& parameter = 1.0);

template <typename T, typename Compare>
std::vector<std::size_t> SortPermutation(const std::vector<T>& vec, Compare compare) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
void ApplyPermutationInPlace(std::vector<T>& vec, const std::vector<std::size_t>& p) {
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

}  // namespace utility
}  // namespace neat
