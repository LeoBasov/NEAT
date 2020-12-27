#pragma once

#include <memory>

#include "genome.h"

namespace neat {
namespace mutator_algorithms {

template <typename Type>
Type Select(const std::vector<Type>& vector, Random& random) {
    return vector.at(random.RandomIntNumber(0, vector.size() - 1));
}

void PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq);

}  // namespace mutator_algorithms
}  // namespace neat
