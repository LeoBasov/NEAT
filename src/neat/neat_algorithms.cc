#include "neat_algorithms.h"

namespace neat {
namespace neat_algorithms {

bool AddNode(genome::Genotype& genotype, GenePool& pool, const uint& gene_id, const double& weight) {
    for (auto& gene : genotype.genes) {
        if (gene.id == gene_id) {
            std::pair<unsigned int, unsigned int> retval = pool.AddNode(gene_id);

            gene.enabled = false;

            genotype.nodes.push_back(pool.GetGene(retval.first).out_node);
            genotype.genes.push_back(genome::Gene(retval.first, 1.0));
            genotype.genes.push_back(genome::Gene(retval.second, weight));

            return true;
        }
    }

    return false;
}

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
