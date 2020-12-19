#pragma once

#include <gtest/gtest.h>

#include "../../../src/common/random_fake.h"
#include "../../../src/neat_new/species_pool.h"

namespace neat {

TEST(SpeciesPool, SortInSpecies) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const double weight1(0.0), weight2(10.0);
    SpeciesPool::Config config;
    SpeciesPool pool;
    std::vector<Genome> genomes{Genome(n_sensor_nodes, n_output_nodes), Genome(n_sensor_nodes, n_output_nodes)};
    std::vector<SpeciesPool::Species> species;

    for (uint i = 0; i < genomes.at(0).genes_.size(); i++) {
        genomes.at(0).genes_.at(i).weight = weight1;
    }

    for (uint i = 0; i < genomes.at(1).genes_.size(); i++) {
        genomes.at(1).genes_.at(i).weight = weight2;
    }

    config.distance_coefficients.at(2) = 1.0;
    config.max_species_distance = 1.0;

    pool.SetConfig(config);
    pool.SortInSpecies(genomes);

    species = pool.GetSpecies();

    ASSERT_EQ(2, species.size());

    ASSERT_EQ(1, species.at(0).n_member);
    ASSERT_EQ(1, species.at(1).n_member);

    ASSERT_EQ(0, genomes.at(0).species_id_);
    ASSERT_EQ(1, genomes.at(1).species_id_);

    pool.SortInSpecies(genomes);

    species = pool.GetSpecies();

    ASSERT_EQ(2, species.size());

    ASSERT_EQ(1, species.at(0).n_member);
    ASSERT_EQ(1, species.at(1).n_member);

    ASSERT_EQ(0, genomes.at(0).species_id_);
    ASSERT_EQ(1, genomes.at(1).species_id_);
}

TEST(SpeciesPool, AdjustFitnesses) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const double weight1(1.0), weight2(10.0);
    SpeciesPool::Config config;
    SpeciesPool pool;
    std::vector<Genome> genomes{Genome(n_sensor_nodes, n_output_nodes), Genome(n_sensor_nodes, n_output_nodes),
                                Genome(n_sensor_nodes, n_output_nodes)};
    std::vector<SpeciesPool::Species> species;
    const std::vector<double> fitnesses_old{4.0, 8.0, 3.0};
    std::vector<double> fitnesses(fitnesses_old);

    for (uint i = 0; i < genomes.at(0).genes_.size(); i++) {
        genomes.at(0).genes_.at(i).weight = weight1;
    }

    for (uint i = 0; i < genomes.at(1).genes_.size(); i++) {
        genomes.at(1).genes_.at(i).weight = weight1;
    }

    for (uint i = 0; i < genomes.at(2).genes_.size(); i++) {
        genomes.at(2).genes_.at(i).weight = weight2;
    }

    config.distance_coefficients.at(2) = 1.0;
    config.max_species_distance = 1.0;

    pool.SetConfig(config);
    pool.SortInSpecies(genomes);

    species = pool.GetSpecies();

    ASSERT_EQ(2, species.size());

    ASSERT_EQ(2, species.at(0).n_member);
    ASSERT_EQ(1, species.at(1).n_member);

    ASSERT_EQ(0, genomes.at(0).species_id_);
    ASSERT_EQ(0, genomes.at(1).species_id_);
    ASSERT_EQ(1, genomes.at(2).species_id_);

    pool.AdjustFitnesses(fitnesses, genomes);

    species = pool.GetSpecies();

    ASSERT_DOUBLE_EQ(fitnesses_old.at(0) / 2.0 + fitnesses_old.at(1) / 2.0, species.at(0).total_fitness);
    ASSERT_DOUBLE_EQ(fitnesses_old.at(2), species.at(1).total_fitness);

    ASSERT_DOUBLE_EQ(fitnesses_old.at(0) / 2.0, fitnesses.at(0));
    ASSERT_DOUBLE_EQ(fitnesses_old.at(1) / 2.0, fitnesses.at(1));
    ASSERT_DOUBLE_EQ(fitnesses_old.at(2), fitnesses.at(2));

    ASSERT_DOUBLE_EQ(fitnesses.at(0) + fitnesses.at(1) + fitnesses.at(2), pool.GetTotalFitness());
}

TEST(SpeciesPool, Clear) {
    const uint n_sensor_nodes(2), n_output_nodes(1);
    const double weight1(0.0), weight2(10.0);
    SpeciesPool::Config config;
    SpeciesPool pool;
    std::vector<Genome> genomes{Genome(n_sensor_nodes, n_output_nodes), Genome(n_sensor_nodes, n_output_nodes)};
    std::vector<SpeciesPool::Species> species;

    for (uint i = 0; i < genomes.at(0).genes_.size(); i++) {
        genomes.at(0).genes_.at(i).weight = weight1;
    }

    for (uint i = 0; i < genomes.at(1).genes_.size(); i++) {
        genomes.at(1).genes_.at(i).weight = weight2;
    }

    config.distance_coefficients.at(2) = 1.0;
    config.max_species_distance = 1.0;

    pool.SetConfig(config);
    pool.SortInSpecies(genomes);

    pool.Clear();

    ASSERT_EQ(0, pool.GetSpecies().size());
    ASSERT_DOUBLE_EQ(0.0, pool.GetTotalFitness());
}

}  // namespace neat
