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
}

}  // namespace neat
