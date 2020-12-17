#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat_new/genome.h"

namespace neat {

TEST(Genome, Constructor) {
    const uint n_sensor_nodes(2), n_output_nodes(3);
    Genome genome(n_sensor_nodes, n_output_nodes);

    ASSERT_EQ(n_sensor_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(0, genome.n_hidden_nodes_);
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 1, genome.nodes_.size());
    ASSERT_EQ((n_sensor_nodes + 1) * n_output_nodes, genome.genes_.size());
}

TEST(Genome, Clear) {
    const uint n_sensor_nodes(2), n_output_nodes(3);
    Genome genome(n_sensor_nodes, n_output_nodes);

    genome.Clear();

    ASSERT_EQ(0, genome.n_sensor_nodes_);
    ASSERT_EQ(0, genome.n_output_nodes_);
    ASSERT_EQ(0, genome.n_hidden_nodes_);
    ASSERT_EQ(0, genome.nodes_.size());
    ASSERT_EQ(0, genome.genes_.size());
}

TEST(Genome, Initialize) {
    const uint n_sensor_nodes(2), n_output_nodes(3);
    Genome genome(n_sensor_nodes, n_output_nodes);

    genome.Initialize(n_sensor_nodes, n_output_nodes);

    ASSERT_EQ(n_sensor_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(0, genome.n_hidden_nodes_);
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 1, genome.nodes_.size());
    ASSERT_EQ((n_sensor_nodes + 1) * n_output_nodes, genome.genes_.size());
}

}  // namespace neat
