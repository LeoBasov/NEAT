#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat.h"

namespace NEAT {

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

}  // namespace NEAT
