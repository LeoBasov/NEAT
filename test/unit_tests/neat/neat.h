#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat.h"

namespace NEAT {
using namespace Eigen;

TEST(NEAT, Initialize) {
    NEAT neat;
    NEAT::Config config;
    const uint n_input(3), n_output(2), n_phenotypes(100);

    config.n_input = n_input;
    config.n_output = n_output;
    config.n_phenotypes = n_phenotypes;

    neat.Initialize(config);

    ASSERT_EQ(n_phenotypes, neat.phenotypes_.size());

    ASSERT_EQ(0, neat.gene_pool_.input_nodes_.ofset);
    ASSERT_EQ(n_input, neat.gene_pool_.input_nodes_.n_parts);

    ASSERT_EQ(n_input, neat.gene_pool_.output_nodes_.ofset);
    ASSERT_EQ(n_output, neat.gene_pool_.output_nodes_.n_parts);

    ASSERT_EQ(n_input + n_output, neat.gene_pool_.hidden_nodes_.ofset);
    ASSERT_EQ(0, neat.gene_pool_.hidden_nodes_.n_parts);

    ASSERT_EQ(n_input + n_output, neat.gene_pool_.nodes_.size());
    ASSERT_EQ(n_input * n_output, neat.gene_pool_.genes_.size());
}

TEST(NEAT, BuildNetworks) {
    NEAT neat;
    NEAT::Config config;

    config.n_input = 3;
    config.n_output = 2;
    config.n_phenotypes = 100;

    neat.Initialize(config);
    neat.BuildNetworks();

    ASSERT_EQ(config.n_phenotypes, neat.networks_.size());
}

TEST(NEAT, Execute) {
    NEAT neat;
    NEAT::Config config;
    VectorXd input(3);
    VectorXd output(2);
    std::vector<std::pair<VectorXd, VectorXd>> input_outputs;

    input_outputs.push_back({input, output});

    config.n_input = 3;
    config.n_output = 2;
    config.n_phenotypes = 100;

    neat.Initialize(config);
    neat.BuildNetworks();
    neat.Execute(input_outputs);
}

TEST(NEAT, AddNode) {
    NEAT neat;
    NEAT::Config config;
    VectorXd input(3);
    VectorXd output(2);
    std::vector<std::pair<VectorXd, VectorXd>> input_outputs;

    input_outputs.push_back({input, output});

    config.n_input = 3;
    config.n_output = 2;
    config.n_phenotypes = 1;

    neat.Initialize(config);

    ASSERT_FALSE(neat.AddNode(0, 3, 2));
    ASSERT_FALSE(neat.AddNode(0, 3, 4));
    ASSERT_FALSE(neat.AddNode(0, 1, 2));

    ASSERT_TRUE(neat.AddNode(0, 2, 3));

    neat.BuildNetworks();
    neat.Execute(input_outputs);
}

}  // namespace NEAT
