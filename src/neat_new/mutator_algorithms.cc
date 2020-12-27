#include "mutator_algorithms.h"

namespace neat {
namespace mutator_algorithms {

void PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq) {
    genome.genes_.at(gene_id).weight *= (1.0 - perturbation_fraq) + 2 * (perturbation_fraq)*random.RandomNumber();
}

double RandomizeWeight(const double& min, const double& max, Random& random) {
    return min + (max - min) * random.RandomNumber();
}

std::pair<bool, uint> InLastGenes(const uint& in, const uint& out, const std::vector<LastGene>& last_genes,
                                  LastGene::Type type) {
    for (uint i = 0; i < last_genes.size(); i++) {
        if ((in == last_genes.at(i).in) && (out == last_genes.at(i).out) && (last_genes.at(i).type == type)) {
            return {true, i};
        }
    }

    return {false, 0};
}

}  // namespace mutator_algorithms
}  // namespace neat
