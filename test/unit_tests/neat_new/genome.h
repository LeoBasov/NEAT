#pragma once

#include <gtest/gtest.h>

#include "../../../src/common/random_fake.h"
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
    const uint innov_begin(((n_sensor_nodes + 1) * n_output_nodes) - 1);
    uint innov(innov_begin);
    Genome genome(n_sensor_nodes, n_output_nodes);
    bool allow_self_connection(false), allow_recurring_connection(false);

    innov = genome.AddNode(gene_id, innov);
    innov = genome.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection);
    ASSERT_EQ(innov_begin + 3, innov);

    innov = genome.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection);
    ASSERT_EQ(innov_begin + 3, innov);

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

    innov = genome1.AddNode(gene_id, innov);
    innov = genome1.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection);

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

    innov = genome1.AddNode(gene_id, innov);
    innov = genome1.AddConnection(in, out, innov, allow_self_connection, allow_recurring_connection);

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

TEST(Genome, AdjustNodes) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    Genome genome;

    genome.genes_ = {Genome::Gene(0, 3, 0), Genome::Gene(1, 3, 1), Genome::Gene(2, 1, 2)};

    genome.AdjustNodes(n_sensor_nodes, n_output_nodes);

    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 1, genome.nodes_.size());

    ASSERT_EQ(n_sensor_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(0, genome.n_hidden_nodes_);

    for (uint i = 0; i < genome.nodes_.size(); i++) {
        ASSERT_EQ(i, genome.nodes_.at(i));
    }

    genome.genes_.push_back(Genome::Gene(0, 4, 3));
    genome.genes_.push_back(Genome::Gene(4, 3, 4));

    genome.AdjustNodes(n_sensor_nodes, n_output_nodes);

    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    ASSERT_EQ(n_sensor_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(1, genome.n_hidden_nodes_);

    for (uint i = 0; i < genome.nodes_.size(); i++) {
        ASSERT_EQ(i, genome.nodes_.at(i));
    }
}

TEST(Genome, Mate) {
    RandomFake random;
    const uint n_sensor_nodes(2), n_output_nodes(1);
    Genome genome1(n_sensor_nodes, n_output_nodes), genome2(n_sensor_nodes, n_output_nodes), child;
    const double weight(13.11);

    genome1.genes_.push_back(Genome::Gene(0, 4, 3));
    genome1.genes_.push_back(Genome::Gene(4, 3, 4));

    genome1.AdjustNodes(n_sensor_nodes, n_output_nodes);

    for (uint i = 0; i < genome2.genes_.size(); i++) {
        genome2.genes_.at(i).weight = weight;
    }

    child = Genome::Mate(genome1, genome2, random);

    ASSERT_EQ(n_sensor_nodes, child.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, child.n_output_nodes_);
    ASSERT_EQ(1, child.n_hidden_nodes_);
    ASSERT_EQ(genome1.nodes_.size(), child.nodes_.size());

    for (uint i = 0; i < child.nodes_.size(); i++) {
        ASSERT_EQ(i, child.nodes_.at(i));
    }

    for (uint i = 0; i < genome2.genes_.size(); i++) {
        ASSERT_DOUBLE_EQ(weight, child.genes_.at(i).weight);
    }

    random.SetRetVal(1.0);

    child = Genome::Mate(genome1, genome2, random);

    ASSERT_EQ(n_sensor_nodes, child.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, child.n_output_nodes_);
    ASSERT_EQ(1, child.n_hidden_nodes_);
    ASSERT_EQ(genome1.nodes_.size(), child.nodes_.size());

    for (uint i = 0; i < child.nodes_.size(); i++) {
        ASSERT_EQ(i, child.nodes_.at(i));
    }

    for (uint i = 0; i < genome2.genes_.size(); i++) {
        ASSERT_DOUBLE_EQ(1.0, child.genes_.at(i).weight);
    }
}

TEST(Genome, GetNodePermuationMap) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    Genome genome(n_sensor_nodes, n_output_nodes);
    std::map<size_t, size_t> map;

    genome.nodes_.push_back(12);
    genome.nodes_.push_back(25);
    genome.nodes_.push_back(30);

    map = genome.GetNodePermuationMap();

    ASSERT_EQ(0, map.at(0));
    ASSERT_EQ(1, map.at(1));
    ASSERT_EQ(2, map.at(2));
    ASSERT_EQ(3, map.at(3));

    ASSERT_EQ(4, map.at(12));
    ASSERT_EQ(5, map.at(25));
    ASSERT_EQ(6, map.at(30));

    genome.nodes_.push_back(30);

    ASSERT_THROW(genome.GetNodePermuationMap(), std::domain_error);
}

}  // namespace neat
