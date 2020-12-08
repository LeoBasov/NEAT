#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/gene_pool.h"

namespace neat {

TEST(GenePool, Initialize) {
    GenePool gene_pool;
    const uint n_sensors = 3, n_output = 2;

    gene_pool.Initialize(n_sensors, n_output);

    ASSERT_EQ(n_sensors, gene_pool.GetNSensorNodes());
    ASSERT_EQ(n_output, gene_pool.GetNOutputNodes());
    ASSERT_EQ(0, gene_pool.GetNHiddenNodes());
    ASSERT_EQ(n_sensors + n_output + 1, gene_pool.GetNTotalNodes());

    ASSERT_EQ((n_sensors + 1) * n_output, gene_pool.GetGenes().size());

    gene_pool.Initialize(n_sensors, n_output);

    ASSERT_EQ(n_sensors, gene_pool.GetNSensorNodes());
    ASSERT_EQ(n_output, gene_pool.GetNOutputNodes());
    ASSERT_EQ(0, gene_pool.GetNHiddenNodes());
    ASSERT_EQ(n_sensors + n_output + 1, gene_pool.GetNTotalNodes());

    ASSERT_EQ((n_sensors + 1) * n_output, gene_pool.GetGenes().size());
}

TEST(GenePool, Clear) {
    GenePool gene_pool;
    const uint n_sensors = 3, n_output = 2;

    gene_pool.Initialize(n_sensors, n_output);
    gene_pool.Clear();

    ASSERT_EQ(0, gene_pool.GetNSensorNodes());
    ASSERT_EQ(0, gene_pool.GetNOutputNodes());
    ASSERT_EQ(0, gene_pool.GetNHiddenNodes());
    ASSERT_EQ(1, gene_pool.GetNTotalNodes());

    ASSERT_EQ(0, gene_pool.GetGenes().size());
}

TEST(GenePool, AddNode) {
    GenePool gene_pool;
    const uint n_sensors = 2, n_output = 1;
    std::pair<unsigned int, unsigned int> retval;

    gene_pool.Initialize(n_sensors, n_output);
    retval = gene_pool.AddNode(0);

    ASSERT_EQ(n_sensors, gene_pool.GetNSensorNodes());
    ASSERT_EQ(n_output, gene_pool.GetNOutputNodes());
    ASSERT_EQ(1, gene_pool.GetNHiddenNodes());
    ASSERT_EQ(n_sensors + n_output + 2, gene_pool.GetNTotalNodes());

    ASSERT_EQ((n_sensors + 1) * n_output + 2, gene_pool.GetGenes().size());

    ASSERT_EQ(gene_pool.GetGenes().at(0).in_node, gene_pool.GetGenes().at(gene_pool.GetGenes().size() - 2).in_node);
    ASSERT_EQ(gene_pool.GetNTotalNodes() - 1, gene_pool.GetGenes().at(gene_pool.GetGenes().size() - 2).out_node);

    ASSERT_EQ(gene_pool.GetNTotalNodes() - 1, gene_pool.GetGenes().at(gene_pool.GetGenes().size() - 1).in_node);
    ASSERT_EQ(gene_pool.GetGenes().at(0).out_node, gene_pool.GetGenes().at(gene_pool.GetGenes().size() - 1).out_node);
}

}  // namespace neat
