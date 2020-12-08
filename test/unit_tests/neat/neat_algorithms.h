#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat_algorithms.h"
#include "neat.h"

namespace neat {
namespace neat_algorithms {

void ExecuteXOR(const MatrixXd& matrix, const GenePool& pool, const double& val1, const double& val2,
                const double& ref) {
    VectorXd vec = neat_algorithms::SetUpNodes({val1, val2}, pool);

    neat_algorithms::ExecuteNetwork(matrix, vec, 3, 5.9);
    neat_algorithms::ExecuteNetwork(matrix, vec, 3, 5.9);

    ASSERT_NEAR(ref, vec(3), 1e-8);
}

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

TEST(neat_algorithms, SetUpNodes) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    std::vector<double> input_values{3.0, 5.0};

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    VectorXd input_vector = SetUpNodes(input_values, gene_pool);

    ASSERT_EQ(4, input_vector.rows());

    ASSERT_DOUBLE_EQ(1.0, input_vector(0));
    ASSERT_DOUBLE_EQ(input_values.at(0), input_vector(1));
    ASSERT_DOUBLE_EQ(input_values.at(1), input_vector(2));
    ASSERT_DOUBLE_EQ(0.0, input_vector(3));
}

TEST(neat_algorithms, ExecuteNetwork) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 1, n_output = 1, n_genotypes = 1;
    std::vector<double> input_values{0.0};

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    genotype.genes.at(0).weight = 0.0;
    genotype.genes.at(1).weight = 1.0;

    MatrixXd network = Genotype2Phenotype(genotype, gene_pool);
    VectorXd nodes = SetUpNodes(input_values, gene_pool);

    ExecuteNetwork(network, nodes, 2);

    ASSERT_EQ(3, nodes.rows());

    ASSERT_DOUBLE_EQ(1.0, nodes(0));
    ASSERT_DOUBLE_EQ(input_values.at(0), nodes(1));
    ASSERT_DOUBLE_EQ(0.5, nodes(2));
}

TEST(neat_algorithms, XOR) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    const double weight(1.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    neat_algorithms::AddNode(genotype, gene_pool, 1, weight);
    neat_algorithms::AddNode(genotype, gene_pool, 2, weight);

    neat_algorithms::AddConnection(genotype, gene_pool, 0, 4, weight);
    neat_algorithms::AddConnection(genotype, gene_pool, 0, 5, weight);

    neat_algorithms::AddConnection(genotype, gene_pool, 1, 5, weight);
    neat_algorithms::AddConnection(genotype, gene_pool, 2, 4, weight);

    // Set BIAS node (2)
    genotype.genes.at(7).weight = -2.32161229;
    genotype.genes.at(8).weight = -5.2368337;
    genotype.genes.at(0).weight = -3.13762134;

    // Set input_node1
    genotype.genes.at(3).weight = 5.70223616;
    genotype.genes.at(9).weight = 3.42762429;

    // Set input_node2
    genotype.genes.at(10).weight = 5.73141813;
    genotype.genes.at(5).weight = 3.4327536;

    // Set hidden nodes
    genotype.genes.at(4).weight = 7.05553511;
    genotype.genes.at(6).weight = -7.68450564;

    MatrixXd matrix = neat_algorithms::Genotype2Phenotype(genotype, gene_pool);

    ExecuteXOR(matrix, gene_pool, 1.0, 1.0, 0.0);
    ExecuteXOR(matrix, gene_pool, 0.0, 0.0, 0.0);
    ExecuteXOR(matrix, gene_pool, 1.0, 0.0, 1.0);
    ExecuteXOR(matrix, gene_pool, 0.0, 1.0, 1.0);
}

}  // namespace neat_algorithms
}  // namespace neat
