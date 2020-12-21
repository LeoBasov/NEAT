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

TEST(Network, GenomeToVector) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const std::vector<double> input_vals{3.0, 7.0};
    Genome genome(n_sensor_nodes, n_output_nodes);
    uint innov(genome.genes_.size());
    VectorXd vector;

    genome.AddNode(0, innov);

    vector = Network::GenomeToVector(input_vals, genome.nodes_.size());

    for (uint i = 0; i < genome.nodes_.size(); i++) {
        if (i == 0) {
            ASSERT_DOUBLE_EQ(1.0, vector(i));
        } else if (i == 1) {
            ASSERT_DOUBLE_EQ(input_vals.at(0), vector(i));
        } else if (i == 2) {
            ASSERT_DOUBLE_EQ(input_vals.at(1), vector(i));
        } else {
            ASSERT_DOUBLE_EQ(0.0, vector(i));
        }
    }
}

TEST(Network, VectorToStd) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const std::vector<double> input_vals{3.0, 7.0};
    Genome genome(n_sensor_nodes, n_output_nodes);
    uint innov(genome.genes_.size());
    VectorXd vector;
    const std::vector<size_t> output_nodes{n_sensor_nodes + 1};

    genome.AddNode(0, innov);

    vector = Network::GenomeToVector(input_vals, genome.nodes_.size());

    vector(n_sensor_nodes + 1) = 123.45;

    static std::vector<double> ret_vec = Network::VectorToStd(vector, output_nodes);

    ASSERT_EQ(n_output_nodes, ret_vec.size());
    ASSERT_DOUBLE_EQ(vector(n_sensor_nodes + 1), ret_vec.front());
}

TEST(Network, Build) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const std::vector<double> input_vals{3.0, 7.0};
    Genome genome(n_sensor_nodes, n_output_nodes);
    uint innov(genome.genes_.size());
    VectorXd vector;
    const std::vector<size_t> output_nodes{n_sensor_nodes + 1};

    genome.AddNode(0, innov);

    Network network(genome);

    ASSERT_EQ(genome.n_hidden_nodes_ + 1, network.n_executions_);
    ASSERT_EQ(genome.n_sensor_nodes_ + 1, network.n_const_nodes_);
}

TEST(Network, Execute) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const std::vector<double> input_vals{3.0, 7.0};
    Genome genome(n_sensor_nodes, n_output_nodes);
    uint innov(genome.genes_.size());

    innov = genome.AddNode(1, innov);
    innov = genome.AddNode(2, innov);

    innov = genome.AddConnection(0, 4, innov);
    innov = genome.AddConnection(0, 5, innov);

    innov = genome.AddConnection(1, 5, innov);
    innov = genome.AddConnection(2, 4, innov);

    // Set BIAS node (2)
    genome.genes_.at(7).weight = -2.32161229;
    genome.genes_.at(8).weight = -5.2368337;
    genome.genes_.at(0).weight = -3.13762134;

    // Set input_node1
    genome.genes_.at(3).weight = 5.70223616;
    genome.genes_.at(9).weight = 3.42762429;

    // Set input_node2
    genome.genes_.at(10).weight = 5.73141813;
    genome.genes_.at(5).weight = 3.4327536;

    // Set hidden nodes
    genome.genes_.at(4).weight = 7.05553511;
    genome.genes_.at(6).weight = -7.68450564;

    Network network(genome);

    ASSERT_NEAR(1.0, network.Execute({0.0, 1.0}).front(), 1e-8);
    ASSERT_NEAR(1.0, network.Execute({1.0, 0.0}).front(), 1e-8);

    ASSERT_NEAR(0.0, network.Execute({1.0, 1.0}).front(), 1e-8);
    ASSERT_NEAR(0.0, network.Execute({0.0, 0.0}).front(), 1e-8);
}

}  // namespace neat
