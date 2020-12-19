#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat_new/neat.h"

namespace neat {

TEST(Neat, Initialize) {
    const uint n_sensort_nodes(2), n_output_nodes(3), n_genomes(150);
    Neat::Config config;
    Neat neat;
    std::vector<Genome> genomes;
    double last_weight;

    config.mutator_config.weight_min = 50.0;
    config.mutator_config.weight_max = 75.0;

    neat.Initialize(n_sensort_nodes, n_output_nodes, n_genomes, config);

    genomes = neat.GetGenomes();

    ASSERT_EQ(n_genomes, genomes.size());

    for (uint i = 0; i < n_genomes; i++) {
        ASSERT_EQ(n_sensort_nodes + n_output_nodes + 1, genomes.at(i).nodes_.size());
        ASSERT_EQ((n_sensort_nodes + 1) * n_output_nodes, genomes.at(i).genes_.size());

        for (uint k = 0; k < genomes.at(i).genes_.size(); k++) {
            if (i || k) {
                ASSERT_FALSE(last_weight == genomes.at(i).genes_.at(k).weight);
            }

            ASSERT_TRUE(genomes.at(i).genes_.at(k).weight <= config.mutator_config.weight_max);
            ASSERT_TRUE(genomes.at(i).genes_.at(k).weight >= config.mutator_config.weight_min);

            last_weight = genomes.at(i).genes_.at(k).weight;
        }
    }
}

}  // namespace neat
