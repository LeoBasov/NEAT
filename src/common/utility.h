#pragma once

#include <algorithm>
#include <climits>
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

struct greater {
    template <class T>
    bool operator()(T const& a, T const& b) const {
        return a > b;
    }
};

template <typename T>
T swap_endian(T u) {
    static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

    union {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++) dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}

}  // namespace utility
}  // namespace neat
