#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat_algorithms.h"
#include "neat.h"

namespace neat {
namespace neat_algorithms {

TEST(neat_algorithms, AddNode) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint gene_id = 0;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    const double new_weight(15.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    ASSERT_FALSE(AddNode(genotype, gene_pool, 4, new_weight));
    ASSERT_TRUE(AddNode(genotype, gene_pool, gene_id, new_weight));

    ASSERT_FALSE(genotype.genes.at(gene_id).enabled);
    ASSERT_EQ(5, genotype.genes.size());
    ASSERT_EQ(3, genotype.genes.at(3).id);
    ASSERT_EQ(4, genotype.genes.at(4).id);
    ASSERT_DOUBLE_EQ(1.0, genotype.genes.at(3).weight);
    ASSERT_DOUBLE_EQ(new_weight, genotype.genes.at(4).weight);

    ASSERT_EQ(0, gene_pool.GetGene(3).in_node);
    ASSERT_EQ(4, gene_pool.GetGene(3).out_node);

    ASSERT_EQ(4, gene_pool.GetGene(4).in_node);
    ASSERT_EQ(3, gene_pool.GetGene(4).out_node);
}

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
