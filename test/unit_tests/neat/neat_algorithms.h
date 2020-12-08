#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat_algorithms.h"
#include "neat.h"

namespace neat {
namespace neat_algorithms {

TEST(neat_algorithms, AddConnection) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint in_node = 3, out_node = 3;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    const double new_weight(15.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    ASSERT_TRUE(AddConnection(genotype, gene_pool, in_node, out_node, new_weight));
    ASSERT_EQ(4, genotype.nodes.size());
    ASSERT_EQ(4, genotype.genes.size());
    ASSERT_DOUBLE_EQ(new_weight, genotype.genes.back().weight);

    GenePool::Gene gene = gene_pool.GetGene(genotype.genes.back().id);

    ASSERT_EQ(in_node, gene.in_node);
    ASSERT_EQ(out_node, gene.out_node);

    ASSERT_FALSE(AddConnection(genotype, gene_pool, in_node, 0, new_weight));
}

}  // namespace neat_algorithms
}  // namespace neat
