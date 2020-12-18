#pragma once

#include <gtest/gtest.h>

#include "../../../src/common/random_fake.h"
#include "../../../src/neat_new/mutator.h"

namespace neat {

TEST(Mutator, PertubateWeight) {
    const uint n_sensor_nodes(2), n_output_nodes(1), gene_id(0);
    const double perturbation_fraction(0.1), weight_pos(100.0), weight_neg(-100.0), weight_null(0.0);
    RandomFake random;
    Genome genome(n_sensor_nodes, n_output_nodes);

    // positive weight
    genome.genes_.at(gene_id).weight = weight_pos;
    random.SetRetVal(0.0);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_pos * (1.0 - perturbation_fraction), genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_pos;
    random.SetRetVal(0.5);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_pos, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_pos;
    random.SetRetVal(1.0);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_pos * (1.0 + perturbation_fraction), genome.genes_.at(gene_id).weight);

    // negative weight
    genome.genes_.at(gene_id).weight = weight_neg;
    random.SetRetVal(0.0);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_neg * (1.0 - perturbation_fraction), genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_neg;
    random.SetRetVal(0.5);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_neg, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_neg;
    random.SetRetVal(1.0);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(weight_neg * (1.0 + perturbation_fraction), genome.genes_.at(gene_id).weight);

    // null weight
    genome.genes_.at(gene_id).weight = weight_null;
    random.SetRetVal(0.0);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(0.0, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_null;
    random.SetRetVal(0.5);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(0.0, genome.genes_.at(gene_id).weight);

    genome.genes_.at(gene_id).weight = weight_null;
    random.SetRetVal(1.0);
    Mutator::PertubateWeight(genome, random, gene_id, perturbation_fraction);
    ASSERT_DOUBLE_EQ(0.0, genome.genes_.at(gene_id).weight);
}

}  // namespace neat
