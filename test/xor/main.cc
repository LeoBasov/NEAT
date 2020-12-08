#include <iostream>

#include "../../src/neat/neat.h"
#include "../../src/neat/neat_algorithms.h"

using namespace neat;

void Execute(const MatrixXd& matrix, const GenePool& pool, const double& val1, const double& val2);

int main(int, char**) {
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

    Execute(matrix, gene_pool, 1.0, 1.0);
    Execute(matrix, gene_pool, 0.0, 0.0);
    Execute(matrix, gene_pool, 1.0, 0.0);
    Execute(matrix, gene_pool, 0.0, 1.0);

    return 0;
}

void Execute(const MatrixXd& matrix, const GenePool& pool, const double& val1, const double& val2) {
    VectorXd vec = neat_algorithms::SetUpNodes({val1, val2}, pool);

    neat_algorithms::ExecuteNetwork(matrix, vec, 3, 5.9);
    neat_algorithms::ExecuteNetwork(matrix, vec, 3, 5.9);

    std::cout << "VAL1: " << val1 << " VAL2: " << val2 << " OUTPUT: " << vec(3) << std::endl;
}
