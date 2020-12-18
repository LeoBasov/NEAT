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

TEST(Mutator, Mutate) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    std::shared_ptr<RandomFake> random(std::make_shared<RandomFake>());
    Genome genome(n_sensor_nodes, n_output_nodes);
    Mutator mutator;
    Mutator::Config config;
    const uint innovation_init(genome.genes_.size());
    uint innovation(innovation_init);

    // NULL
    config.prob_new_node = 0.5;
    config.prob_new_connection = 0.0;
    config.prob_new_weight = 0.0;
    config.prob_weight_change = 0.0;

    random->SetRetVal(1.0);

    mutator.SetConfig(config);
    mutator.SetrRandom(random);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init, innovation);

    // AddNode
    config.prob_new_node = 0.5;
    config.prob_new_connection = 0.0;
    config.prob_new_weight = 0.0;
    config.prob_weight_change = 0.0;

    random->SetRetVal(0.4);

    mutator.SetConfig(config);
    mutator.SetrRandom(random);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init + 2, innovation);
    ASSERT_EQ(innovation, genome.genes_.size());
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    // AddConecction
    config.prob_new_node = 0.0;
    config.prob_new_connection = 0.5;
    config.prob_new_weight = 0.0;
    config.prob_weight_change = 0.0;

    random->SetRetVal(0.4);
    random->SetRetValInt(0);

    mutator.SetConfig(config);
    mutator.SetrRandom(random);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init + 2, innovation);
    ASSERT_EQ(innovation, genome.genes_.size());
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    random->SetRetValInt(3);

    mutator.SetrRandom(random);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init + 3, innovation);
    ASSERT_EQ(innovation, genome.genes_.size());
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    ASSERT_EQ(3, genome.genes_.back().in);
    ASSERT_EQ(3, genome.genes_.back().out);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init + 3, innovation);
    ASSERT_EQ(innovation, genome.genes_.size());
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    ASSERT_EQ(3, genome.genes_.back().in);
    ASSERT_EQ(3, genome.genes_.back().out);

    // Assign new weight
    config.prob_new_node = 0.0;
    config.prob_new_connection = 0.0;
    config.prob_new_weight = 0.5;
    config.prob_weight_change = 0.5;

    random->SetRetVal(0.4);
    random->SetRetValInt(0);

    mutator.SetConfig(config);
    mutator.SetrRandom(random);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init + 3, innovation);
    ASSERT_EQ(innovation, genome.genes_.size());
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    ASSERT_DOUBLE_EQ(config.weight_min + (config.weight_max - config.weight_min) * 0.4, genome.genes_.at(0).weight);

    // Perturbate new weight
    config.prob_new_node = 0.0;
    config.prob_new_connection = 0.0;
    config.prob_new_weight = 0.0;
    config.prob_weight_change = 1.1;

    random->SetRetVal(1.0);
    random->SetRetValInt(0);

    mutator.SetConfig(config);
    mutator.SetrRandom(random);

    const double weight(genome.genes_.at(0).weight);

    mutator.Mutate(genome, innovation);

    ASSERT_EQ(innovation_init + 3, innovation);
    ASSERT_EQ(innovation, genome.genes_.size());
    ASSERT_EQ(n_sensor_nodes + n_output_nodes + 2, genome.nodes_.size());

    ASSERT_DOUBLE_EQ(weight * 1.1, genome.genes_.at(0).weight);
}

}  // namespace neat
