#pragma once

#include <gtest/gtest.h>

#include "../../../src/io/genome.h"

namespace neat {

TEST(genome, ReadGenome) {
    const uint n_sensort_nodes(2), n_output_nodes(1), n_hidden_nodes(1), n_genes(7);
    const std::string file_name("./test/unit_tests/test_data/genome_2_1.csv");
    Genome genome;

    ASSERT_THROW(genome::ReadGenome("", n_sensort_nodes, n_output_nodes), Exception);

    genome = genome::ReadGenome(file_name, n_sensort_nodes, n_output_nodes);

    ASSERT_EQ(n_sensort_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(n_hidden_nodes, genome.n_hidden_nodes_);
    ASSERT_EQ(n_genes, genome.genes_.size());

    ASSERT_EQ(0, genome.genes_.at(0).in);
    ASSERT_EQ(3, genome.genes_.at(0).out);
    ASSERT_EQ(0, genome.genes_.at(0).innov);
    ASSERT_EQ(-421.906, genome.genes_.at(0).weight);

    ASSERT_EQ(1, genome.genes_.at(1).in);
    ASSERT_EQ(3, genome.genes_.at(1).out);
    ASSERT_EQ(1, genome.genes_.at(1).innov);
    ASSERT_EQ(501.628, genome.genes_.at(1).weight);

    ASSERT_EQ(2, genome.genes_.at(6).in);
    ASSERT_EQ(4, genome.genes_.at(6).out);
    ASSERT_EQ(6, genome.genes_.at(6).innov);
    ASSERT_EQ(545.51, genome.genes_.at(6).weight);
}

}  // namespace neat
