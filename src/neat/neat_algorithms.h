#pragma once

#include <algorithm>

#include "../common/random.h"
#include "gene_pool.h"
#include "genome.h"

namespace neat {
namespace neat_algorithms {

bool AddNode(genome::Genotype& genotype, GenePool& pool, const uint& gene_id, const double& weight);
bool AddConnection(genome::Genotype& genotype, GenePool& pool, const uint& in_node, const uint& out_node,
                   const double& weight);
genome::Genotype Mate(const genome::Genotype& fitter_parent, const genome::Genotype& parent, Random& random);
double CalcDistance(const std::vector<genome::Gene>& genome1, const std::vector<genome::Gene>& genome2,
                    const double& ceff1, const double& ceff2, const double& ceff3);

}  // namespace neat_algorithms
}  // namespace neat
