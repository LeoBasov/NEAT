#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat.h"

namespace NEAT {
using namespace Eigen;

TEST(NEAT, Initialize) {
    NEAT neat;
    NEAT::Config config;

    config.n_input = 3;
    config.n_output = 2;
    config.n_phenotypes = 100;

    neat.Initialize(config);
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

}  // namespace NEAT
