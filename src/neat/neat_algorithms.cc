#include "neat_algorithms.h"

namespace neat {
namespace neat_algorithms {

bool AddConnection(genome::Genotype& genotype, GenePool& pool, const uint& in_node, const uint& out_node,
                   const double& weight) {
    std::pair<bool, unsigned int> retval = pool.AddConnection(in_node, out_node);

    if (retval.first) {
        if (std::find(genotype.genes.begin(), genotype.genes.end(), retval.second) != genotype.genes.end()) {
            return false;
        } else {
            genotype.genes.push_back(genome::Gene(retval.second, weight));
        }
    }

    return retval.first;
}

}  // namespace neat_algorithms
}  // namespace neat
