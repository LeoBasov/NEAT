#pragma once

#include <memory>

#include "genome.h"

namespace neat {
namespace mutator_algorithms {

template <typename Type>
size_t SelectId(const std::vector<Type>& vector, Random& random) {
    return random.RandomIntNumber(0, vector.size() - 1);
}

template <typename Type>
Type Select(const std::vector<Type>& vector, Random& random) {
    return vector.at(SelectId(vector, random));
}

void PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq);
double RandomizeWeight(const double& min, const double& max, Random& random);

}  // namespace mutator_algorithms
}  // namespace neat
