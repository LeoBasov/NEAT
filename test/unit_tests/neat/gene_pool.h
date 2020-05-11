#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/gene_pool.h"

namespace NEAT {

TEST(GenePool, Initialize) {
    GenePool pool;

    pool.Initialize(3, 2);

    ASSERT_EQ(5, pool.nodes_.size());

    ASSERT_EQ(0, pool.input_nodes_.ofset);
    ASSERT_EQ(3, pool.input_nodes_.n_parts);

    ASSERT_EQ(3, pool.output_nodes_.ofset);
    ASSERT_EQ(2, pool.output_nodes_.n_parts);

    ASSERT_EQ(5, pool.hidden_nodes_.ofset);
    ASSERT_EQ(0, pool.hidden_nodes_.n_parts);

    ASSERT_EQ(6, pool.genes_.size());

    for (uint i = pool.input_nodes_.ofset; i < pool.input_nodes_.ofset + pool.input_nodes_.n_parts; i++) {
        ASSERT_EQ(0, pool.nodes_.at(i).level);
    }

    for (uint i = pool.output_nodes_.ofset; i < pool.output_nodes_.ofset + pool.output_nodes_.n_parts; i++) {
        ASSERT_EQ(1, pool.nodes_.at(i).level);
    }
}

TEST(GenePool, AddNode) {
    GenePool pool;

    pool.Initialize(3, 2);

    ASSERT_FALSE(pool.AddNode(0, 1));
    ASSERT_FALSE(pool.AddNode(0, 2));
    ASSERT_FALSE(pool.AddNode(2, 1));

    ASSERT_FALSE(pool.AddNode(3, 4));

    ASSERT_FALSE(pool.AddNode(3, 0));
    ASSERT_FALSE(pool.AddNode(3, 1));
    ASSERT_FALSE(pool.AddNode(3, 2));

    ASSERT_FALSE(pool.AddNode(4, 0));
    ASSERT_FALSE(pool.AddNode(4, 1));
    ASSERT_FALSE(pool.AddNode(4, 2));

    ASSERT_TRUE(pool.AddNode(0, 3));

    ASSERT_EQ(0, pool.nodes_.at(0).level);
    ASSERT_EQ(0, pool.nodes_.at(1).level);
    ASSERT_EQ(0, pool.nodes_.at(2).level);

    ASSERT_EQ(2, pool.nodes_.at(3).level);
    ASSERT_EQ(2, pool.nodes_.at(4).level);

    ASSERT_EQ(1, pool.nodes_.at(5).level);

    ASSERT_EQ(0, pool.genes_.at(6).in);
    ASSERT_EQ(5, pool.genes_.at(6).out);

    ASSERT_EQ(5, pool.genes_.at(7).in);
    ASSERT_EQ(3, pool.genes_.at(7).out);
}

TEST(GenePool, AddConnection) {
    GenePool pool;

    pool.Initialize(3, 2);

    ASSERT_FALSE(pool.AddConnection(0, 1).first);
    ASSERT_FALSE(pool.AddConnection(0, 2).first);
    ASSERT_FALSE(pool.AddConnection(2, 1).first);

    ASSERT_FALSE(pool.AddConnection(3, 4).first);

    ASSERT_FALSE(pool.AddConnection(3, 0).first);
    ASSERT_FALSE(pool.AddConnection(3, 1).first);
    ASSERT_FALSE(pool.AddConnection(3, 2).first);

    ASSERT_FALSE(pool.AddConnection(4, 0).first);
    ASSERT_FALSE(pool.AddConnection(4, 1).first);
    ASSERT_FALSE(pool.AddConnection(4, 2).first);

    pool.AddNode(0, 3);

    ASSERT_TRUE(pool.AddConnection(1, 5).first);

    ASSERT_EQ(1, pool.genes_.at(8).in);
    ASSERT_EQ(5, pool.genes_.at(8).out);
}

TEST(GenePool, Clear) {
    GenePool pool;

    pool.Initialize(3, 2);
    pool.AddNode(0, 3);
    pool.AddConnection(1, 5);

    pool.Clear();

    ASSERT_EQ(0, pool.input_nodes_.ofset);
    ASSERT_EQ(0, pool.input_nodes_.n_parts);
    ASSERT_EQ(0, pool.output_nodes_.ofset);
    ASSERT_EQ(0, pool.output_nodes_.n_parts);
    ASSERT_EQ(0, pool.hidden_nodes_.ofset);
    ASSERT_EQ(0, pool.hidden_nodes_.n_parts);
    ASSERT_EQ(0, pool.nodes_.size());
    ASSERT_EQ(0, pool.genes_.size());
}

}  // namespace NEAT
