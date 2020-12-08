#pragma once

#include <algorithm>

#include "gene_pool.h"
#include "genome.h"

namespace neat {
namespace neat_algorithms {

bool AddNode(genome::Genotype& genotype, GenePool& pool, const uint& gene_id, const double& weight);
bool AddConnection(genome::Genotype& genotype, GenePool& pool, const uint& in_node, const uint& out_node,
                   const double& weight);

}  // namespace neat_algorithms
}  // namespace neat
