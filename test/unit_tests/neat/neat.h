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

TEST(NEAT, Sigmoid) {
    NEAT neat;
    const double input_1(-1000.0), input_2(0.0), input_3(1000.0);
    const double output_1(0.0), output_2(0.5), output_3(1.0);

    ASSERT_DOUBLE_EQ(output_1, neat.Sigmoid(input_1));
    ASSERT_DOUBLE_EQ(output_2, neat.Sigmoid(input_2));
    ASSERT_DOUBLE_EQ(output_3, neat.Sigmoid(input_3));
}

TEST(NEAT, Execute) {
    // XOR test
    NEAT neat;
    NEAT::Config config;
    VectorXd input1(3), input2(3), input3(3), input4(3);
    VectorXd output1(1), output2(1), output3(1), output4(1);
    std::vector<std::pair<VectorXd, VectorXd>> input_outputs;

    input1(0) = 1.0;
    input1(1) = 1.0;
    input1(2) = 1.0;

    input2(0) = 1.0;
    input2(1) = 0.0;
    input2(2) = 1.0;

    input3(0) = 1.0;
    input3(1) = 1.0;
    input3(2) = 0.0;

    input4(0) = 1.0;
    input4(1) = 0.0;
    input4(2) = 0.0;

    output1(0) = 0.0;
    output2(0) = 1.0;
    output3(0) = 1.0;
    output4(0) = 0.0;

    input_outputs.push_back({input1, output1});
    input_outputs.push_back({input2, output2});
    input_outputs.push_back({input3, output3});
    input_outputs.push_back({input4, output4});

    config.n_input = 3;
    config.n_output = 1;
    config.n_phenotypes = 1;
    config.sigmoid_parameter = 4.9;

    neat.Initialize(config);

    neat.AddNode(0, 1, 3);
    neat.AddNode(0, 2, 3);

    neat.AddConnection(0, 0, 4);
    neat.AddConnection(0, 0, 5);

    neat.AddConnection(0, 1, 5);
    neat.AddConnection(0, 2, 4);

    // Set BIAS node (2)
    neat.SetWeight(0, 7, -2.32161229);
    neat.SetWeight(0, 8, -5.2368337);
    neat.SetWeight(0, 0, -3.13762134);

    // Set input_node1
    neat.SetWeight(0, 3, 5.70223616);
    neat.SetWeight(0, 9, 3.42762429);

    // Set input_node2
    neat.SetWeight(0, 10, 5.73141813);
    neat.SetWeight(0, 5, 3.4327536);

    // Set hidden nodes
    neat.SetWeight(0, 4, 7.05553511);
    neat.SetWeight(0, 6, -7.68450564);

    neat.BuildNetworks();
    neat.Execute(input_outputs);

    ASSERT_NEAR(1.0, neat.phenotypes_.at(0).fitness_, 1.0e-5);
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
