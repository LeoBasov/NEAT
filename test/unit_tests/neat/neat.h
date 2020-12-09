#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat.h"

namespace neat {

TEST(NEAT, Initialize) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 3, n_output = 2, n_genotypes = 10;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    ASSERT_EQ(n_sensors, gene_pool.GetNSensorNodes());
    ASSERT_EQ(n_output, gene_pool.GetNOutputNodes());
    ASSERT_EQ(0, gene_pool.GetNHiddenNodes());
    ASSERT_EQ(n_sensors + n_output + 1, gene_pool.GetNTotalNodes());

    ASSERT_EQ(0, gene_pool.GetGene(0).in_node);
    ASSERT_EQ(4, gene_pool.GetGene(0).out_node);
    ASSERT_EQ(0, gene_pool.GetGene(1).in_node);
    ASSERT_EQ(5, gene_pool.GetGene(1).out_node);
    ASSERT_EQ(1, gene_pool.GetGene(2).in_node);
    ASSERT_EQ(4, gene_pool.GetGene(2).out_node);

    ASSERT_EQ((n_sensors + 1) * n_output, gene_pool.GetGenes().size());

    ASSERT_EQ(n_genotypes, genotypes.size());
    ASSERT_EQ(gene_pool.GetGenes().size(), genotypes.front().genes.size());
    ASSERT_EQ(gene_pool.GetNTotalNodes(), genotypes.front().nodes.size());

    for (uint i = 0; i < genotypes.front().genes.size(); i++) {
        ASSERT_EQ(i, genotypes.front().genes.at(i).id);
        ASSERT_TRUE(genotypes.front().genes.at(i).enabled);
    }
}

TEST(NEAT, Clear) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 3, n_output = 2, n_genotypes = 10;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    neat.Clear();

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    gene_pool.Initialize(n_sensors, n_output);
    gene_pool.Clear();

    ASSERT_EQ(0, gene_pool.GetNSensorNodes());
    ASSERT_EQ(0, gene_pool.GetNOutputNodes());
    ASSERT_EQ(0, gene_pool.GetNHiddenNodes());
    ASSERT_EQ(1, gene_pool.GetNTotalNodes());

    ASSERT_EQ(0, gene_pool.GetGenes().size());
    ASSERT_EQ(0, genotypes.size());
}

TEST(NEAT, ExecuteNetwork) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 10;
    const std::vector<double> input{3.0, 5.0};
    std::vector<double> output;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    genotypes = neat.GetGenotypes();

    genotypes.at(0).genes.at(0).weight = 0.0;
    genotypes.at(0).genes.at(1).weight = 0.0;
    genotypes.at(0).genes.at(2).weight = 1.0;

    neat.SetGenotypes(genotypes);

    output = neat.ExecuteNetwork(input, 0);

    ASSERT_EQ(1, output.size());

    ASSERT_DOUBLE_EQ(utility::Sigmoid(input.at(1), config.sigmoid_parameter), output.front());

    genotypes.at(0).genes.at(0).weight = 1.0;
    genotypes.at(0).genes.at(1).weight = 1.0;
    genotypes.at(0).genes.at(2).weight = 1.0;

    neat.SetGenotypes(genotypes);

    output = neat.ExecuteNetwork(input, 0);

    ASSERT_EQ(1, output.size());

    ASSERT_DOUBLE_EQ(utility::Sigmoid(1.0 + input.at(0) + input.at(1), config.sigmoid_parameter), output.front());
}

TEST(NEAT, ExecuteNetworks) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 10;
    const std::vector<double> input{3.0, 5.0};
    std::vector<std::vector<double>> output;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    genotypes = neat.GetGenotypes();

    genotypes.at(0).genes.at(0).weight = 0.0;
    genotypes.at(0).genes.at(1).weight = 0.0;
    genotypes.at(0).genes.at(2).weight = 1.0;

    neat.SetGenotypes(genotypes);

    output = neat.ExecuteNetworks(input);

    ASSERT_EQ(10, output.size());
    ASSERT_EQ(1, output.front().size());

    ASSERT_DOUBLE_EQ(utility::Sigmoid(input.at(1), config.sigmoid_parameter), output.front().front());

    genotypes.at(0).genes.at(0).weight = 1.0;
    genotypes.at(0).genes.at(1).weight = 1.0;
    genotypes.at(0).genes.at(2).weight = 1.0;

    neat.SetGenotypes(genotypes);

    output = neat.ExecuteNetworks(input);

    ASSERT_EQ(10, output.size());
    ASSERT_EQ(1, output.front().size());

    ASSERT_DOUBLE_EQ(utility::Sigmoid(1.0 + input.at(0) + input.at(1), config.sigmoid_parameter),
                     output.front().front());
}

TEST(NEAT, UpdateNetworks) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    std::vector<genome::Species> species;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 10;
    const std::vector<double> input{3.0, 5.0};
    std::vector<double> fitnesses(n_genotypes, 1.0);

    config.prob_mate = 0.0;

    config.prob_new_connection = 0.0;
    config.prob_new_node = 0.0;
    config.prob_new_weight = 0.0;
    config.prob_weight_change = 0.0;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    genotypes = neat.GetGenotypes();
    species = neat.GetSpecies();

    for (uint i = 0; i < genotypes.size(); i++) {
        for (uint j = 0; j < genotypes.at(i).genes.size(); j++) {
            genotypes.at(i).genes.at(j).weight = 1.0;
            genotypes.at(i).species_id = 0;
        }
    }

    species.pop_back();
    species.front().n_memeber = n_genotypes;
    species.front().total_fitness = 0.0;
    species.front().ref_genotype = genotypes.front();

    neat.SetGenotypes(genotypes);
    neat.SetSpecies(species);

    neat.UpdateNetworks(fitnesses);

    genotypes = neat.GetGenotypes();
    species = neat.GetSpecies();

    ASSERT_EQ(n_genotypes, genotypes.size());
    ASSERT_EQ(1, species.size());

    for (auto spec : neat.GetSpecies()) {
        ASSERT_EQ(n_genotypes, spec.n_memeber);
    }
}

}  // namespace neat
