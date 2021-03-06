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

TEST(NEAT, UpdateNetworks_NULL) {
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

    species.resize(1);
    species.front().n_member = n_genotypes;
    species.front().total_fitness = 0.0;
    species.front().ref_genotype = genotypes.front();

    neat.SetGenotypes(genotypes);
    neat.SetSpecies(species);

    neat.UpdateNetworks(fitnesses);

    genotypes = neat.GetGenotypes();
    species = neat.GetSpecies();

    ASSERT_EQ(n_genotypes, genotypes.size());
    ASSERT_EQ(1, species.size());

    for (auto spec : species) {
        ASSERT_EQ(n_genotypes, spec.n_member);
    }

    for (auto genotype : genotypes) {
        ASSERT_EQ(0, genotype.species_id);
        ASSERT_EQ(3, genotype.genes.size());
        ASSERT_EQ(4, genotype.nodes.size());

        for (auto gene : genotype.genes) {
            ASSERT_DOUBLE_EQ(1.0, gene.weight);
            ASSERT_TRUE(gene.enabled);
        }
    }
}

TEST(NEAT, UpdateNetworks_Mate_1) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    std::vector<genome::Species> species;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 10;
    const std::vector<double> input{3.0, 5.0};
    std::vector<double> fitnesses(n_genotypes, 1.0);

    config.prob_mate = 1.0;

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

    species.resize(1);
    species.front().n_member = n_genotypes;
    species.front().total_fitness = 0.0;
    species.front().ref_genotype = genotypes.front();

    neat.SetGenotypes(genotypes);
    neat.SetSpecies(species);

    neat.UpdateNetworks(fitnesses);

    genotypes = neat.GetGenotypes();
    species = neat.GetSpecies();

    ASSERT_EQ(n_genotypes, genotypes.size());
    ASSERT_EQ(1, species.size());

    for (auto spec : species) {
        ASSERT_EQ(n_genotypes, spec.n_member);
    }

    for (auto genotype : genotypes) {
        ASSERT_EQ(0, genotype.species_id);
        ASSERT_EQ(3, genotype.genes.size());
        ASSERT_EQ(4, genotype.nodes.size());

        for (auto gene : genotype.genes) {
            ASSERT_DOUBLE_EQ(1.0, gene.weight);
            ASSERT_TRUE(gene.enabled);
        }
    }
}

TEST(NEAT, UpdateNetworks_Mate_2) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    std::vector<genome::Species> species;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 10;
    const std::vector<double> input{3.0, 5.0};
    std::vector<double> fitnesses(n_genotypes, 1.0);

    config.prob_mate = 1.0;

    config.coeff1 = 12.0;

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

    species.resize(1);
    species.front().n_member = n_genotypes;
    species.front().total_fitness = 0.0;
    species.front().ref_genotype = genotypes.front();

    gene_pool = neat.GetGenePool();
    neat_algorithms::AddConnection(genotypes.front(), gene_pool, 3, 3, 1.0, true, true);
    genotypes.front().genes.front().weight = 10.0;
    fitnesses.front() = 10;

    neat.SetGenePool(gene_pool);
    neat.SetGenotypes(genotypes);
    neat.SetSpecies(species);

    neat.UpdateNetworks(fitnesses);

    genotypes = neat.GetGenotypes();
    species = neat.GetSpecies();

    ASSERT_EQ(n_genotypes, genotypes.size());
    ASSERT_EQ(2, species.size());

    ASSERT_EQ(n_genotypes - 1, species.at(0).n_member);
    ASSERT_EQ(1, species.at(1).n_member);

    for (uint i = 1; i < genotypes.size(); i++) {
        ASSERT_EQ(0, genotypes.at(i).species_id);
        ASSERT_EQ(3, genotypes.at(i).genes.size());
        ASSERT_EQ(4, genotypes.at(i).nodes.size());

        for (auto gene : genotypes.at(i).genes) {
            ASSERT_DOUBLE_EQ(1.0, gene.weight);
            ASSERT_TRUE(gene.enabled);
        }
    }

    ASSERT_EQ(1, genotypes.at(0).species_id);
    ASSERT_EQ(4, genotypes.at(0).genes.size());
    ASSERT_EQ(4, genotypes.at(0).nodes.size());
}

}  // namespace neat
