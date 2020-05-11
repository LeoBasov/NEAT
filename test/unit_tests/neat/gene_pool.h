#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/gene_pool.h"

namespace NEAT {

TEST(GenePool, Initialize) {
    GenePool pool;

    pool.Initialize(3, 2);

    ASSERT_EQ(2, pool.depth_);

    ASSERT_EQ(5, pool.nodes_.size());

    ASSERT_EQ(0, pool.node_ofset_.at(0));
    ASSERT_EQ(3, pool.node_ofset_.at(1));
    ASSERT_EQ(5, pool.node_ofset_.at(2));

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
}

}  // namespace NEAT
