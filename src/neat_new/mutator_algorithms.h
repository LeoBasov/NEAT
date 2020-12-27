#pragma once

#include <memory>

#include "genome.h"

namespace neat {
namespace mutator_algorithms {

struct LastGene {
    enum Type { ADD_NODE, ADD_CONNECTION, NONE };

    Type type = NONE;
    uint in = 0, out = 0;
    std::pair<Genome::Gene, Genome::Gene> genes;
};

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
std::pair<bool, uint> InLastGenes(const uint& in, const uint& out, const std::vector<LastGene>& last_genes,
                                  LastGene::Type type);
uint AdjustLastGenes(Genome& genome, std::vector<LastGene>& last_genes, const uint& gene_id, const uint& innovation,
                     const uint& last_innovation);

}  // namespace mutator_algorithms
}  // namespace neat
