#pragma once

#include <gtest/gtest.h>

#include "../../../src/common/random_fake.h"
#include "../../../src/neat_new/mutator_algorithms.h"

namespace neat {

using namespace mutator_algorithms;

TEST(mutator_algorithms, SelectId) {
    std::vector<uint> values;
    std::vector<bool> found(4, false);
    Random random;

    values.push_back(5);
    values.push_back(6);
    values.push_back(7);
    values.push_back(8);

    for (uint i = 0; i < 100; i++) {
        size_t id = SelectId(values, random);
        uint val = values.at(id);

        ASSERT_TRUE(val <= 8);
        ASSERT_TRUE(val >= 5);

        for (uint i = 0; i < values.size(); i++) {
            if (val == values.at(i)) {
                found.at(i) = true;
            }
        }
    }

    bool found_val(true);
    for (auto f : found) {
        found_val = found_val && f;
    }

    ASSERT_TRUE(found_val);
}

TEST(mutator_algorithms, Select) {
    std::vector<uint> values;
    std::vector<bool> found(4, false);
    Random random;

    values.push_back(5);
    values.push_back(6);
    values.push_back(7);
    values.push_back(8);

    for (uint i = 0; i < 100; i++) {
        uint val = Select(values, random);

        ASSERT_TRUE(val <= 8);
        ASSERT_TRUE(val >= 5);

        for (uint i = 0; i < values.size(); i++) {
            if (val == values.at(i)) {
                found.at(i) = true;
            }
        }
    }

    bool found_val(true);
    for (auto f : found) {
        found_val = found_val && f;
    }

    ASSERT_TRUE(found_val);
}

TEST(mutator_algorithms, PertubateWeight) {
    const uint n_sensor_nodes(2), n_output_nodes(1), gene_id(0);
    const double perturbation_fraction(0.1), weight_pos(100.0), weight_neg(-100.0), weight_null(0.0);
    RandomFake random;
    Genome genome(n_sensor_nodes, n_output_nodes);

    // positive weight
    genome.genes_.at(gene_id).weight = weight_pos;
    random.SetRetVal(0.0);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_pos * (1.0 - perturbation_fraction), genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_pos;
    random.SetRetVal(0.5);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_pos, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_pos;
    random.SetRetVal(1.0);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_pos * (1.0 + perturbation_fraction), genome.genes_.at(gene_id).weight);

    // negative weight
    genome.genes_.at(gene_id).weight = weight_neg;
    random.SetRetVal(0.0);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_neg * (1.0 - perturbation_fraction), genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_neg;
    random.SetRetVal(0.5);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_neg, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_neg;
    random.SetRetVal(1.0);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_neg * (1.0 + perturbation_fraction), genome.genes_.at(gene_id).weight);

    // null weight
    genome.genes_.at(gene_id).weight = weight_null;
    random.SetRetVal(0.0);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(0.0, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_null;
    random.SetRetVal(0.5);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(0.0, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_null;
    random.SetRetVal(1.0);
    PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(0.0, genome.genes_.at(gene_id).weight);
}

TEST(mutator_algorithms, RandomizeWeight) {
    const double min(-3.1), max(7.3);
    RandomFake random;

    random.SetRetVal(0.0);
    ASSERT_DOUBLE_EQ(min, RandomizeWeight(min, max, random));

    random.SetRetVal(1.0);
    ASSERT_DOUBLE_EQ(max, RandomizeWeight(min, max, random));

    random.SetRetVal(0.5);
    ASSERT_DOUBLE_EQ((min + max) * 0.5, RandomizeWeight(min, max, random));
}

TEST(mutator_algorithms, InLastGenes) {
    const uint n_sensor_nodes(1), n_output_nodes(1);
    Genome genome(n_sensor_nodes, n_output_nodes);
    std::vector<LastGene> last_genes;
    LastGene last_gene;
    std::pair<bool, uint> ret_pair;

    last_gene.in = 1;
    last_gene.out = 2;
    last_gene.type = LastGene::ADD_NODE;
    last_gene.genes.first = Genome::Gene(1, 3, 2);
    last_gene.genes.first = Genome::Gene(3, 2, 3);

    last_genes.push_back(last_gene);

    genome.AddNode(1, 1);

    ret_pair = InLastGenes(1, 2, last_genes, LastGene::ADD_NODE);

    ASSERT_TRUE(ret_pair.first);
    ASSERT_EQ(0, ret_pair.second);

    ret_pair = InLastGenes(0, 2, last_genes, LastGene::ADD_NODE);

    ASSERT_FALSE(ret_pair.first);
}

TEST(mutator_algorithms, AdjustLastGenes) {
    const uint n_sensor_nodes(1), n_output_nodes(1);
    Genome genome1(n_sensor_nodes, n_output_nodes), genome2(n_sensor_nodes, n_output_nodes);
    std::vector<LastGene> last_genes;
    uint innovation(2), innovation_old, innovation_ref;
    const uint gene_id(0);

    innovation_old = innovation;
    innovation = genome1.AddNode(gene_id, innovation);
    innovation_ref = innovation;
    innovation = AdjustLastGenes(genome1, last_genes, gene_id, innovation, innovation_old);

    ASSERT_EQ(1, last_genes.size());
    ASSERT_EQ(innovation_ref, innovation);

    innovation_old = innovation;
    innovation = genome2.AddNode(gene_id, innovation);
    innovation_ref = innovation;
    innovation = AdjustLastGenes(genome2, last_genes, gene_id, innovation, innovation_old);

    ASSERT_EQ(1, last_genes.size());
    ASSERT_EQ(innovation_old, innovation);

    ASSERT_EQ(genome1.genes_.size(), genome2.genes_.size());
    ASSERT_EQ(genome1.nodes_.size(), genome2.nodes_.size());

    for (uint i = 0; i < genome1.genes_.size(); i++) {
        ASSERT_EQ(genome1.genes_.at(i).innov, genome2.genes_.at(i).innov);
        ASSERT_EQ(genome1.nodes_.at(i), genome2.nodes_.at(i));
    }
}

}  // namespace neat
