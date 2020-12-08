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

TEST(neat_algorithms, Mate) {
    NEAT::Config config;
    genome::Genotype genotype1, genotype2;
    GenePool gene_pool;
    NEAT neat;
    Random random;
    const uint in_node = 3, out_node = 3, gene_id = 0;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 2;
    const double new_weight(15.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype1 = neat.GetGenotypes().at(0);
    genotype2 = neat.GetGenotypes().at(1);

    AddNode(genotype1, gene_pool, gene_id, new_weight);
    AddConnection(genotype2, gene_pool, in_node, out_node, new_weight);

    genome::Genotype child = Mate(genotype1, genotype2, random);

    ASSERT_EQ(genotype1.nodes, child.nodes);
    ASSERT_EQ(genotype1.genes.size(), child.genes.size());
}

TEST(neat_algorithms, CalcDistance) {
    genome::Genotype genotype1, genotype2;

    genotype1.genes.push_back(genome::Gene(0, 1.0));
    genotype1.genes.push_back(genome::Gene(1, 2.0));
    genotype1.genes.push_back(genome::Gene(2, 3.0));
    genotype1.genes.push_back(genome::Gene(3, 6.0));

    genotype2.genes.push_back(genome::Gene(0, 4.0));
    genotype2.genes.push_back(genome::Gene(1, 5.0));
    genotype2.genes.push_back(genome::Gene(3, 6.0));
    genotype2.genes.push_back(genome::Gene(5, 7.0));
    genotype2.genes.push_back(genome::Gene(6, 7.0));

    ASSERT_EQ(0.0, CalcDistance(genotype1.genes, genotype1.genes, 1.0, 1.0, 1.0));
    ASSERT_EQ(0.0, CalcDistance(genotype2.genes, genotype2.genes, 1.0, 1.0, 1.0));

    ASSERT_EQ(2.6, CalcDistance(genotype1.genes, genotype2.genes, 1.0, 1.0, 1.0));
    ASSERT_EQ(2.6, CalcDistance(genotype2.genes, genotype1.genes, 1.0, 1.0, 1.0));
}

TEST(neat_algorithms, Genotype2Phenotype) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 1, n_output = 1, n_genotypes = 1;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    for (auto& gene : genotype.genes) {
        gene.weight = 1.0;
    }

    MatrixXd network = Genotype2Phenotype(genotype, gene_pool);

    ASSERT_EQ(3, network.cols());
    ASSERT_EQ(3, network.rows());

    ASSERT_DOUBLE_EQ(1.0, network(0, 0));
    ASSERT_DOUBLE_EQ(0.0, network(0, 1));
    ASSERT_DOUBLE_EQ(0.0, network(0, 2));

    ASSERT_DOUBLE_EQ(0.0, network(1, 0));
    ASSERT_DOUBLE_EQ(1.0, network(1, 1));
    ASSERT_DOUBLE_EQ(0.0, network(1, 2));

    ASSERT_DOUBLE_EQ(1.0, network(2, 0));
    ASSERT_DOUBLE_EQ(1.0, network(2, 1));
    ASSERT_DOUBLE_EQ(0.0, network(2, 2));
}

}  // namespace neat_algorithms
}  // namespace neat
