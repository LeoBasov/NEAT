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

    for (uint i = 0; i < genome.genes_.size(); i++) {
        ASSERT_EQ(i, genome.genes_.at(i).innov);
        ASSERT_TRUE(genome.genes_.at(i).enabled);
    }
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

    for (uint i = 0; i < genome.genes_.size(); i++) {
        ASSERT_EQ(i, genome.genes_.at(i).innov);
        ASSERT_TRUE(genome.genes_.at(i).enabled);
    }
}

TEST(Genome, AddNode) {
    const uint n_sensor_nodes(2), n_output_nodes(3), gene_id(0), innov(((n_sensor_nodes + 1) * n_output_nodes) - 1);
    const double weight(1375.0);
    Genome genome(n_sensor_nodes, n_output_nodes);

    genome.genes_.at(gene_id).weight = weight;

    genome.AddNode(gene_id, innov);

    ASSERT_EQ(n_sensor_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(1, genome.n_hidden_nodes_);
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());
    ASSERT_EQ(((n_sensor_nodes + 1) * n_output_nodes) + 2, genome.genes_.size());

    ASSERT_DOUBLE_EQ(1.0, genome.genes_.at(genome.genes_.size() - 2).weight);
    ASSERT_DOUBLE_EQ(weight, genome.genes_.at(genome.genes_.size() - 1).weight);

    for (uint i = 0; i < genome.genes_.size(); i++) {
        ASSERT_EQ(i, genome.genes_.at(i).innov);
        ASSERT_TRUE(genome.genes_.at(i).enabled);
    }
}

}  // namespace neat
