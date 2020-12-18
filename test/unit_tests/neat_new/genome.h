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

        if (i == gene_id) {
            ASSERT_FALSE(genome.genes_.at(i).enabled);
        } else {
            ASSERT_TRUE(genome.genes_.at(i).enabled);
        }
    }
}

TEST(Genome, AddConnection) {
    const uint n_sensor_nodes(2), n_output_nodes(1), gene_id(0), in(1), out(4);
    uint innov(((n_sensor_nodes + 1) * n_output_nodes) - 1);
    Genome genome(n_sensor_nodes, n_output_nodes);
    bool allow_self_connection(false), allow_recurring_connection(false);

    genome.AddNode(gene_id, innov);
    innov += 2;

    ASSERT_TRUE(genome.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection));
    ASSERT_FALSE(genome.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection));
    innov++;

    ASSERT_EQ(innov, genome.genes_.size() - 1);

    ASSERT_EQ(in, genome.genes_.back().in);
    ASSERT_EQ(out, genome.genes_.back().out);

    for (uint i = 0; i < genome.genes_.size(); i++) {
        ASSERT_EQ(i, genome.genes_.at(i).innov);

        if (i == gene_id) {
            ASSERT_FALSE(genome.genes_.at(i).enabled);
        } else {
            ASSERT_TRUE(genome.genes_.at(i).enabled);
        }
    }

    ASSERT_EQ(n_sensor_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(1, genome.n_hidden_nodes_);
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());
    ASSERT_EQ(((n_sensor_nodes + 1) * n_output_nodes) + 3, genome.genes_.size());
}

TEST(Genome, Distance_static) {
    const uint n_sensor_nodes(2), n_output_nodes(1), gene_id(0), in(1), out(4);
    const std::array<double, 3> coefficient{1.0, 3.0, 5.0};
    uint innov(((n_sensor_nodes + 1) * n_output_nodes) - 1);
    Genome genome1(n_sensor_nodes, n_output_nodes), genome2(n_sensor_nodes, n_output_nodes);
    bool allow_self_connection(false), allow_recurring_connection(false);

    genome1.AddNode(gene_id, innov);
    innov += 2;

    ASSERT_TRUE(genome1.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection));
    ASSERT_FALSE(genome1.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection));
    innov++;

    ASSERT_DOUBLE_EQ(0.0, Genome::Distance(genome1, genome1, coefficient));
    ASSERT_DOUBLE_EQ(0.0, Genome::Distance(genome2, genome2, coefficient));

    ASSERT_DOUBLE_EQ(0.5, Genome::Distance(genome1, genome2, coefficient));

    genome2.genes_.at(0).weight = 3.0;
    genome2.genes_.at(1).weight = 5.0;
    genome2.genes_.at(2).weight = 7.0;

    ASSERT_DOUBLE_EQ((2.0 + 4.0 + 6.0) * 5.0 / 3.0 + 0.5, Genome::Distance(genome1, genome2, coefficient));

    genome2.genes_.push_back(genome1.genes_.at(4));

    ASSERT_DOUBLE_EQ((2.0 + 4.0 + 6.0) * 5.0 / 4.0 + (1.0 * 1.0 / 6.0) + (3.0 * 1.0 / 6.0),
                     Genome::Distance(genome1, genome2, coefficient));
}

TEST(Genome, Distance) {
    const uint n_sensor_nodes(2), n_output_nodes(1), gene_id(0), in(1), out(4);
    const std::array<double, 3> coefficient{1.0, 3.0, 5.0};
    uint innov(((n_sensor_nodes + 1) * n_output_nodes) - 1);
    Genome genome1(n_sensor_nodes, n_output_nodes), genome2(n_sensor_nodes, n_output_nodes);
    bool allow_self_connection(false), allow_recurring_connection(false);

    genome1.AddNode(gene_id, innov);
    innov += 2;

    ASSERT_TRUE(genome1.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection));
    ASSERT_FALSE(genome1.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection));
    innov++;

    ASSERT_DOUBLE_EQ(0.0, genome1.Distance(genome1, coefficient));
    ASSERT_DOUBLE_EQ(0.0, genome2.Distance(genome2, coefficient));

    ASSERT_DOUBLE_EQ(0.5, genome1.Distance(genome2, coefficient));

    genome2.genes_.at(0).weight = 3.0;
    genome2.genes_.at(1).weight = 5.0;
    genome2.genes_.at(2).weight = 7.0;

    ASSERT_DOUBLE_EQ((2.0 + 4.0 + 6.0) * 5.0 / 3.0 + 0.5, genome1.Distance(genome2, coefficient));

    genome2.genes_.push_back(genome1.genes_.at(4));

    ASSERT_DOUBLE_EQ((2.0 + 4.0 + 6.0) * 5.0 / 4.0 + (1.0 * 1.0 / 6.0) + (3.0 * 1.0 / 6.0),
                     genome1.Distance(genome2, coefficient));
}

}  // namespace neat
