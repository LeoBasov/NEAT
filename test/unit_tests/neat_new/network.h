#pragma once

#include <gtest/gtest.h>
#include <iostream>

#include "../../../src/neat_new/network.h"

namespace neat {

TEST(Network, SetUpOutputNodes) {
    const uint n_sensor_nodes(2), n_output_nodes(3);
    Genome genome(n_sensor_nodes, n_output_nodes);
    uint innov(genome.genes_.size());

    genome.AddNode(0, innov);

    std::vector<size_t> output_nodes = Network::SetUpOutputNodes(genome);

    ASSERT_EQ(n_output_nodes, output_nodes.size());

    for (uint i = 0; i < output_nodes.size(); i++) {
        ASSERT_EQ(output_nodes.at(i), i + genome.n_sensor_nodes_ + 1);
    }
}

TEST(Network, GenomeToMatrix) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    Genome genome(n_sensor_nodes, n_output_nodes);
    uint innov(genome.genes_.size());
    MatrixXd matrix;

    genome.AddNode(0, innov);

    matrix = Network::GenomeToMatrix(genome);

    ASSERT_EQ(genome.nodes_.size(), matrix.rows());
    ASSERT_EQ(genome.nodes_.size(), matrix.cols());

    for (uint i = 0; i < genome.nodes_.size(); i++) {
        for (uint k = 0; k < genome.nodes_.size(); k++) {
            if (i == 3 && k == 1) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else if (i == 3 && k == 2) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else if (i == 4 && k == 0) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else if (i == 3 && k == 4) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else if (i == 0 && k == 0) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else if (i == 1 && k == 1) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else if (i == 2 && k == 2) {
                ASSERT_DOUBLE_EQ(1.0, matrix(i, k));
            } else {
                ASSERT_DOUBLE_EQ(0.0, matrix(i, k));
            }
        }
    }
}

}  // namespace neat
