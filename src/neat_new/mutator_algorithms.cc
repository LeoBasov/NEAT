#include "mutator_algorithms.h"

namespace neat {
namespace mutator_algorithms {

void PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq) {
    genome.genes_.at(gene_id).weight *= (1.0 - perturbation_fraq) + 2 * (perturbation_fraq)*random.RandomNumber();
}

}  // namespace mutator_algorithms
}  // namespace neat
