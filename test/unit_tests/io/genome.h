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

TEST(genome, ReadGenomeRaw) {
    const uint n_sensort_nodes(2), n_output_nodes(1), n_hidden_nodes(1), n_genes(5);
    const std::string file_name("./test/unit_tests/test_data/genome_2_1_raw.csv");
    Genome genome;

    ASSERT_THROW(genome::ReadGenomeRaw("", n_sensort_nodes, n_output_nodes), Exception);

    genome = genome::ReadGenomeRaw(file_name, n_sensort_nodes, n_output_nodes);

    ASSERT_EQ(n_sensort_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(n_hidden_nodes, genome.n_hidden_nodes_);
    ASSERT_EQ(n_genes, genome.genes_.size());

    ASSERT_EQ(0, genome.genes_.at(0).in);
    ASSERT_EQ(3, genome.genes_.at(0).out);
    ASSERT_EQ(0, genome.genes_.at(0).innov);
    ASSERT_DOUBLE_EQ(-421.906, genome.genes_.at(0).weight);
    ASSERT_TRUE(genome.genes_.at(0).enabled);

    ASSERT_EQ(1, genome.genes_.at(1).in);
    ASSERT_EQ(3, genome.genes_.at(1).out);
    ASSERT_EQ(1, genome.genes_.at(1).innov);
    ASSERT_DOUBLE_EQ(501.628, genome.genes_.at(1).weight);
    ASSERT_FALSE(genome.genes_.at(1).enabled);

    ASSERT_EQ(2, genome.genes_.at(2).in);
    ASSERT_EQ(3, genome.genes_.at(2).out);
    ASSERT_EQ(2, genome.genes_.at(2).innov);
    ASSERT_DOUBLE_EQ(605.658, genome.genes_.at(2).weight);
    ASSERT_TRUE(genome.genes_.at(2).enabled);

    ASSERT_EQ(1, genome.genes_.at(3).in);
    ASSERT_EQ(4, genome.genes_.at(3).out);
    ASSERT_EQ(3, genome.genes_.at(3).innov);
    ASSERT_DOUBLE_EQ(13.0, genome.genes_.at(3).weight);
    ASSERT_TRUE(genome.genes_.at(3).enabled);

    ASSERT_EQ(4, genome.genes_.at(4).in);
    ASSERT_EQ(3, genome.genes_.at(4).out);
    ASSERT_EQ(4, genome.genes_.at(4).innov);
    ASSERT_DOUBLE_EQ(17.0, genome.genes_.at(4).weight);
    ASSERT_TRUE(genome.genes_.at(4).enabled);
}

TEST(genome, WriteGenomeRaw) {
    const uint n_sensort_nodes(2), n_output_nodes(1), n_hidden_nodes(1), n_genes(5);
    const std::string file_name("genome_2_1_raw.csv");
    Genome genome_init(n_sensort_nodes, n_output_nodes), genome;

    genome_init.AddNode(1, 2);

    genome_init.genes_.at(0).weight = -421.906;
    genome_init.genes_.at(1).weight = 501.628;
    genome_init.genes_.at(2).weight = 605.658;
    genome_init.genes_.at(3).weight = 13.0;
    genome_init.genes_.at(4).weight = 17.0;

    genome::WriteGenomeRaw(file_name, genome_init);
    genome = genome::ReadGenomeRaw(file_name, n_sensort_nodes, n_output_nodes);

    ASSERT_EQ(n_sensort_nodes, genome.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome.n_output_nodes_);
    ASSERT_EQ(n_hidden_nodes, genome.n_hidden_nodes_);
    ASSERT_EQ(n_genes, genome.genes_.size());

    ASSERT_EQ(0, genome.genes_.at(0).in);
    ASSERT_EQ(3, genome.genes_.at(0).out);
    ASSERT_EQ(0, genome.genes_.at(0).innov);
    ASSERT_DOUBLE_EQ(-421.906, genome.genes_.at(0).weight);
    ASSERT_TRUE(genome.genes_.at(0).enabled);

    ASSERT_EQ(1, genome.genes_.at(1).in);
    ASSERT_EQ(3, genome.genes_.at(1).out);
    ASSERT_EQ(1, genome.genes_.at(1).innov);
    ASSERT_DOUBLE_EQ(501.628, genome.genes_.at(1).weight);
    ASSERT_FALSE(genome.genes_.at(1).enabled);

    ASSERT_EQ(2, genome.genes_.at(2).in);
    ASSERT_EQ(3, genome.genes_.at(2).out);
    ASSERT_EQ(2, genome.genes_.at(2).innov);
    ASSERT_DOUBLE_EQ(605.658, genome.genes_.at(2).weight);
    ASSERT_TRUE(genome.genes_.at(2).enabled);

    ASSERT_EQ(1, genome.genes_.at(3).in);
    ASSERT_EQ(4, genome.genes_.at(3).out);
    ASSERT_EQ(3, genome.genes_.at(3).innov);
    ASSERT_DOUBLE_EQ(13.0, genome.genes_.at(3).weight);
    ASSERT_TRUE(genome.genes_.at(3).enabled);

    ASSERT_EQ(4, genome.genes_.at(4).in);
    ASSERT_EQ(3, genome.genes_.at(4).out);
    ASSERT_EQ(4, genome.genes_.at(4).innov);
    ASSERT_DOUBLE_EQ(17.0, genome.genes_.at(4).weight);
    ASSERT_TRUE(genome.genes_.at(4).enabled);
}

TEST(genome, WriteGenome) {
    const uint n_sensort_nodes(1), n_output_nodes(1), n_hidden_nodes(1), n_genes(4);
    const std::string file_name("genome_2_1.csv");
    Genome genome(n_sensort_nodes, n_output_nodes), genome_read;

    genome.genes_.push_back(Genome::Gene(0, 155, 20));
    genome.genes_.push_back(Genome::Gene(155, 2, 21));
    genome.AdjustNodes(n_sensort_nodes, n_output_nodes);

    genome::WriteGenome(file_name, genome);
    genome_read = genome::ReadGenome(file_name, n_sensort_nodes, n_output_nodes);

    ASSERT_EQ(n_sensort_nodes, genome_read.n_sensor_nodes_);
    ASSERT_EQ(n_output_nodes, genome_read.n_output_nodes_);
    ASSERT_EQ(n_hidden_nodes, genome_read.n_hidden_nodes_);
    ASSERT_EQ(n_genes, genome_read.genes_.size());

    ASSERT_EQ(0, genome_read.genes_.at(0).in);
    ASSERT_EQ(2, genome_read.genes_.at(0).out);
    ASSERT_EQ(0, genome_read.genes_.at(0).innov);
    ASSERT_DOUBLE_EQ(1.0, genome_read.genes_.at(0).weight);
    ASSERT_TRUE(genome_read.genes_.at(0).enabled);

    ASSERT_EQ(1, genome_read.genes_.at(1).in);
    ASSERT_EQ(2, genome_read.genes_.at(1).out);
    ASSERT_EQ(1, genome_read.genes_.at(1).innov);
    ASSERT_DOUBLE_EQ(1.0, genome_read.genes_.at(1).weight);
    ASSERT_TRUE(genome_read.genes_.at(1).enabled);

    ASSERT_EQ(0, genome_read.genes_.at(2).in);
    ASSERT_EQ(3, genome_read.genes_.at(2).out);
    ASSERT_EQ(2, genome_read.genes_.at(2).innov);
    ASSERT_DOUBLE_EQ(1.0, genome_read.genes_.at(2).weight);
    ASSERT_TRUE(genome_read.genes_.at(2).enabled);

    ASSERT_EQ(3, genome_read.genes_.at(3).in);
    ASSERT_EQ(2, genome_read.genes_.at(3).out);
    ASSERT_EQ(3, genome_read.genes_.at(3).innov);
    ASSERT_DOUBLE_EQ(1.0, genome_read.genes_.at(3).weight);
    ASSERT_TRUE(genome_read.genes_.at(3).enabled);
}

}  // namespace neat
