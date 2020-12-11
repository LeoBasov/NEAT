#pragma once

#include <gtest/gtest.h>

#include "../../../src/neat/neat_algorithms.h"
#include "neat.h"

namespace neat {
namespace neat_algorithms {

void ExecuteXOR(const MatrixXd& matrix, const uint& n_nodes, const double& val1, const double& val2,
                const double& ref) {
    VectorXd vec = neat_algorithms::SetUpNodes({val1, val2}, n_nodes);

    neat_algorithms::ExecuteNetwork(matrix, vec, 3, 5.9);
    neat_algorithms::ExecuteNetwork(matrix, vec, 3, 5.9);

    ASSERT_NEAR(ref, vec(3), 1e-8);
}

TEST(neat_algorithms, AddNode) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint gene_id = 0;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    const double new_weight(15.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    ASSERT_FALSE(AddNode(genotype, gene_pool, 4, new_weight));
    ASSERT_TRUE(AddNode(genotype, gene_pool, gene_id, new_weight));

    ASSERT_FALSE(genotype.genes.at(gene_id).enabled);
    ASSERT_EQ(5, genotype.genes.size());
    ASSERT_EQ(3, genotype.genes.at(3).id);
    ASSERT_EQ(4, genotype.genes.at(4).id);
    ASSERT_DOUBLE_EQ(1.0, genotype.genes.at(3).weight);
    ASSERT_DOUBLE_EQ(new_weight, genotype.genes.at(4).weight);

    ASSERT_EQ(0, gene_pool.GetGene(3).in_node);
    ASSERT_EQ(4, gene_pool.GetGene(3).out_node);

    ASSERT_EQ(4, gene_pool.GetGene(4).in_node);
    ASSERT_EQ(3, gene_pool.GetGene(4).out_node);
}

TEST(neat_algorithms, AddConnection) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint in_node = 3, out_node = 3;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    const double new_weight(15.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    ASSERT_TRUE(AddConnection(genotype, gene_pool, in_node, out_node, new_weight, true, true));
    ASSERT_EQ(4, genotype.nodes.size());
    ASSERT_EQ(4, genotype.genes.size());
    ASSERT_DOUBLE_EQ(new_weight, genotype.genes.back().weight);

    GenePool::Gene gene = gene_pool.GetGene(genotype.genes.back().id);

    ASSERT_EQ(in_node, gene.in_node);
    ASSERT_EQ(out_node, gene.out_node);

    ASSERT_FALSE(AddConnection(genotype, gene_pool, in_node, 0, new_weight, true, true));
}

TEST(neat_algorithms, Mate) {
    NEAT::Config config;
    genome::Genotype genotype1, genotype2;
    GenePool gene_pool;
    NEAT neat;
    Random random;
    const uint in_node = 3, out_node = 3, gene_id = 0;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 2;
    const double new_weight(15.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype1 = neat.GetGenotypes().at(0);
    genotype2 = neat.GetGenotypes().at(1);

    AddNode(genotype1, gene_pool, gene_id, new_weight);
    AddConnection(genotype2, gene_pool, in_node, out_node, new_weight, true, true);

    genome::Genotype child = Mate(genotype1, genotype2, random);

    ASSERT_EQ(genotype1.nodes, child.nodes);
    ASSERT_EQ(genotype1.genes.size(), child.genes.size());
}

TEST(neat_algorithms, CalcDistance) {
    genome::Genotype genotype1, genotype2;

    genotype1.genes.push_back(genome::Gene(0, 1.0));
    genotype1.genes.push_back(genome::Gene(1, 2.0));
    genotype1.genes.push_back(genome::Gene(2, 3.0));
    genotype1.genes.push_back(genome::Gene(3, 6.0));

    genotype2.genes.push_back(genome::Gene(0, 4.0));
    genotype2.genes.push_back(genome::Gene(1, 5.0));
    genotype2.genes.push_back(genome::Gene(3, 6.0));
    genotype2.genes.push_back(genome::Gene(5, 7.0));
    genotype2.genes.push_back(genome::Gene(6, 7.0));

    ASSERT_EQ(0.0, CalcDistance(genotype1.genes, genotype1.genes, 1.0, 1.0, 1.0));
    ASSERT_EQ(0.0, CalcDistance(genotype2.genes, genotype2.genes, 1.0, 1.0, 1.0));

    ASSERT_EQ(2.6, CalcDistance(genotype1.genes, genotype2.genes, 1.0, 1.0, 1.0));
    ASSERT_EQ(2.6, CalcDistance(genotype2.genes, genotype1.genes, 1.0, 1.0, 1.0));

    genotype1.genes.clear();
    genotype2.genes.clear();

    genotype1.genes.push_back(genome::Gene(0, 1.0));
    genotype1.genes.push_back(genome::Gene(1, 1.0));
    genotype1.genes.push_back(genome::Gene(2, 1.0));

    genotype2.genes.push_back(genome::Gene(0, 1.0));
    genotype2.genes.push_back(genome::Gene(1, 1.0));
    genotype2.genes.push_back(genome::Gene(2, 1.0));
    genotype2.genes.push_back(genome::Gene(3, 10.0));

    ASSERT_EQ(0.0, CalcDistance(genotype1.genes, genotype1.genes, 1.0, 1.0, 1.0));
    ASSERT_EQ(0.0, CalcDistance(genotype2.genes, genotype2.genes, 1.0, 1.0, 1.0));

    ASSERT_EQ(2.0, CalcDistance(genotype1.genes, genotype2.genes, 8.0, 1.0, 1.0));
    ASSERT_EQ(2.0, CalcDistance(genotype2.genes, genotype1.genes, 8.0, 1.0, 1.0));
}

TEST(neat_algorithms, Genotype2Phenotype) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 1, n_output = 1, n_genotypes = 1;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    for (auto& gene : genotype.genes) {
        gene.weight = 1.0;
    }

    MatrixXd network = Genotype2Phenotype(genotype, gene_pool);

    ASSERT_EQ(3, network.cols());
    ASSERT_EQ(3, network.rows());

    ASSERT_DOUBLE_EQ(1.0, network(0, 0));
    ASSERT_DOUBLE_EQ(0.0, network(0, 1));
    ASSERT_DOUBLE_EQ(0.0, network(0, 2));

    ASSERT_DOUBLE_EQ(0.0, network(1, 0));
    ASSERT_DOUBLE_EQ(1.0, network(1, 1));
    ASSERT_DOUBLE_EQ(0.0, network(1, 2));

    ASSERT_DOUBLE_EQ(1.0, network(2, 0));
    ASSERT_DOUBLE_EQ(1.0, network(2, 1));
    ASSERT_DOUBLE_EQ(0.0, network(2, 2));
}

TEST(neat_algorithms, SetUpNodes) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    std::vector<double> input_values{3.0, 5.0};

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    VectorXd input_vector = SetUpNodes(input_values, genotype.nodes.size());

    ASSERT_EQ(4, input_vector.rows());

    ASSERT_DOUBLE_EQ(1.0, input_vector(0));
    ASSERT_DOUBLE_EQ(input_values.at(0), input_vector(1));
    ASSERT_DOUBLE_EQ(input_values.at(1), input_vector(2));
    ASSERT_DOUBLE_EQ(0.0, input_vector(3));
}

TEST(neat_algorithms, ExecuteNetwork) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 1, n_output = 1, n_genotypes = 1;
    std::vector<double> input_values{0.0};

    neat.Initialize(n_sensors, n_output, n_genotypes, config);
    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    genotype.genes.at(0).weight = 0.0;
    genotype.genes.at(1).weight = 1.0;

    MatrixXd network = Genotype2Phenotype(genotype, gene_pool);
    VectorXd nodes = SetUpNodes(input_values, genotype.nodes.size());

    ExecuteNetwork(network, nodes, 2);

    ASSERT_EQ(3, nodes.rows());

    ASSERT_DOUBLE_EQ(1.0, nodes(0));
    ASSERT_DOUBLE_EQ(input_values.at(0), nodes(1));
    ASSERT_DOUBLE_EQ(0.5, nodes(2));
}

TEST(neat_algorithms, XOR) {
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 1;
    const double weight(1.0);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotype = neat.GetGenotypes().front();

    neat_algorithms::AddNode(genotype, gene_pool, 1, weight);
    neat_algorithms::AddNode(genotype, gene_pool, 2, weight);

    neat_algorithms::AddConnection(genotype, gene_pool, 0, 4, weight, true, true);
    neat_algorithms::AddConnection(genotype, gene_pool, 0, 5, weight, true, true);

    neat_algorithms::AddConnection(genotype, gene_pool, 1, 5, weight, true, true);
    neat_algorithms::AddConnection(genotype, gene_pool, 2, 4, weight, true, true);

    // Set BIAS node (2)
    genotype.genes.at(7).weight = -2.32161229;
    genotype.genes.at(8).weight = -5.2368337;
    genotype.genes.at(0).weight = -3.13762134;

    // Set input_node1
    genotype.genes.at(3).weight = 5.70223616;
    genotype.genes.at(9).weight = 3.42762429;

    // Set input_node2
    genotype.genes.at(10).weight = 5.73141813;
    genotype.genes.at(5).weight = 3.4327536;

    // Set hidden nodes
    genotype.genes.at(4).weight = 7.05553511;
    genotype.genes.at(6).weight = -7.68450564;

    MatrixXd matrix = neat_algorithms::Genotype2Phenotype(genotype, gene_pool);

    ExecuteXOR(matrix, genotype.nodes.size(), 1.0, 1.0, 0.0);
    ExecuteXOR(matrix, genotype.nodes.size(), 0.0, 0.0, 0.0);
    ExecuteXOR(matrix, genotype.nodes.size(), 1.0, 0.0, 1.0);
    ExecuteXOR(matrix, genotype.nodes.size(), 0.0, 1.0, 1.0);
}

TEST(neat_algorithms, SortInSpecies) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    std::vector<genome::Species> species;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 5;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    for (auto& genotype : genotypes) {
        for (auto& gene : genotype.genes) {
            gene.weight = 1.0;
        }
    }

    genotypes.at(0).genes.at(0).weight = -100.0;
    genotypes.at(1).genes.at(0).weight = 100.0;

    neat_algorithms::SortInSpecies(genotypes, species, 3.0, 1.0, 1.0, 0.4);

    ASSERT_EQ(3, species.size());
}

TEST(neat_algorithms, AdjustedFitnesses) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    std::vector<genome::Species> species;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 6;
    std::vector<double> fitnesses;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    for (auto& genotype : genotypes) {
        for (auto& gene : genotype.genes) {
            gene.weight = 1.0;
        }
    }

    genotypes.at(0).genes.at(0).weight = -100.0;
    genotypes.at(1).genes.at(0).weight = -100.0;
    genotypes.at(2).genes.at(0).weight = 100.0;

    neat_algorithms::SortInSpecies(genotypes, species, 3.0, 1.0, 1.0, 0.4);

    ASSERT_EQ(3, species.size());

    ASSERT_THROW(neat_algorithms::AdjustedFitnesses(fitnesses, species, genotypes), std::domain_error);

    fitnesses = {1.0, 6.0, 13.0, 12.0, 12.0, 12.0};

    neat_algorithms::AdjustedFitnesses(fitnesses, species, genotypes);

    ASSERT_DOUBLE_EQ(0.5, fitnesses.at(0));
    ASSERT_DOUBLE_EQ(3.0, fitnesses.at(1));
    ASSERT_DOUBLE_EQ(13.0, fitnesses.at(2));
    ASSERT_DOUBLE_EQ(4.0, fitnesses.at(3));
    ASSERT_DOUBLE_EQ(4.0, fitnesses.at(4));
    ASSERT_DOUBLE_EQ(4.0, fitnesses.at(5));

    // species
    ASSERT_DOUBLE_EQ(3.5, species.at(0).total_fitness);
    ASSERT_DOUBLE_EQ(13.0, species.at(1).total_fitness);
    ASSERT_DOUBLE_EQ(12.0, species.at(2).total_fitness);
}

TEST(neat_algorithms, SortByFitness) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    std::vector<genome::Species> species;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 3;
    std::vector<double> fitnesses;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    for (auto& genotype : genotypes) {
        for (auto& gene : genotype.genes) {
            gene.weight = 1.0;
        }
    }

    SortInSpecies(genotypes, species, 3.0, 1.0, 1.0, 0.4);

    fitnesses = {13.0, 1.0, 7.0};

    AdjustedFitnesses(fitnesses, species, genotypes);

    genotypes.at(0).genes.pop_back();
    genotypes.at(1).nodes.pop_back();

    SortByFitness(fitnesses, genotypes);

    ASSERT_EQ(2, genotypes.at(0).genes.size());
    ASSERT_EQ(4, genotypes.at(0).nodes.size());

    ASSERT_EQ(3, genotypes.at(1).genes.size());
    ASSERT_EQ(4, genotypes.at(1).nodes.size());

    ASSERT_EQ(3, genotypes.at(2).genes.size());
    ASSERT_EQ(3, genotypes.at(2).nodes.size());
};

TEST(neat_algorithms, SortBySpecies) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    std::vector<genome::Species> species;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 6;
    std::vector<double> fitnesses;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    for (auto& genotype : genotypes) {
        for (auto& gene : genotype.genes) {
            gene.weight = 1.0;
        }
    }

    genotypes.at(5).genes.at(0).weight = -100.0;
    genotypes.at(0).genes.at(0).weight = -100.0;
    genotypes.at(2).genes.at(0).weight = 100.0;

    SortInSpecies(genotypes, species, 3.0, 1.0, 1.0, 0.4);

    fitnesses = {1.0, 6.0, 13.0, 12.0, 12.0, 12.0};

    AdjustedFitnesses(fitnesses, species, genotypes);
    SortBySpecies(genotypes);

    ASSERT_EQ(0, genotypes.at(0).species_id);
    ASSERT_EQ(0, genotypes.at(1).species_id);

    ASSERT_EQ(1, genotypes.at(2).species_id);
    ASSERT_EQ(1, genotypes.at(3).species_id);
    ASSERT_EQ(1, genotypes.at(4).species_id);

    ASSERT_EQ(2, genotypes.at(5).species_id);
};

TEST(neat_algorithms, ReproduceSpecies) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes, genotypes_new;
    GenePool gene_pool;
    NEAT neat;
    std::vector<genome::Species> species;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 6;
    std::vector<double> fitnesses;
    const double prob_mate(0.75);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    for (auto& genotype : genotypes) {
        for (auto& gene : genotype.genes) {
            gene.weight = 1.0;
        }
    }

    genotypes.at(5).genes.at(0).weight = -100.0;
    genotypes.at(0).genes.at(0).weight = -100.0;
    genotypes.at(2).genes.at(0).weight = 100.0;

    fitnesses = {1.0, 6.0, 13.0, 12.0, 12.0, 12.0};

    SortInSpecies(genotypes, species, 3.0, 1.0, 1.0, 0.4);

    AdjustedFitnesses(fitnesses, species, genotypes);
    SortByFitness(fitnesses, genotypes);
    SortBySpecies(genotypes);

    ReproduceSpecies(species.at(2), genotypes, genotypes_new, 3, 2, prob_mate);

    ASSERT_EQ(3, genotypes_new.size());
};

TEST(neat_algorithms, Reproduce) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes;
    GenePool gene_pool;
    NEAT neat;
    std::vector<genome::Species> species;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 6;
    std::vector<double> fitnesses;
    const double prob_mate(0.75);

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool = neat.GetGenePool();
    genotypes = neat.GetGenotypes();

    for (auto& genotype : genotypes) {
        for (auto& gene : genotype.genes) {
            gene.weight = 1.0;
        }
    }

    fitnesses = {1.0, 6.0, 13.0, 12.0, 12.0, 12.0};

    SortInSpecies(genotypes, species, 3.0, 1.0, 1.0, 0.4);

    AdjustedFitnesses(fitnesses, species, genotypes);
    SortByFitness(fitnesses, genotypes);
    SortBySpecies(genotypes);

    Reproduce(fitnesses, species, genotypes, 3, prob_mate);

    ASSERT_EQ(3, genotypes.size());
};

TEST(neat_algorithms, Mutate_NULL) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes_new, genotypes_old;
    GenePool gene_pool_new, gene_pool_old;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 6;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool_new = neat.GetGenePool();
    gene_pool_old = neat.GetGenePool();
    genotypes_old = neat.GetGenotypes();
    genotypes_new = neat.GetGenotypes();

    Mutate(genotypes_new, gene_pool_new, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true, true);

    ASSERT_EQ(gene_pool_old.GetNSensorNodes(), gene_pool_new.GetNSensorNodes());
    ASSERT_EQ(gene_pool_old.GetNOutputNodes(), gene_pool_new.GetNOutputNodes());
    ASSERT_EQ(gene_pool_old.GetNHiddenNodes(), gene_pool_new.GetNHiddenNodes());
    ASSERT_EQ(gene_pool_old.GetNTotalNodes(), gene_pool_new.GetNTotalNodes());
    ASSERT_EQ(gene_pool_old.GetGenes().size(), gene_pool_new.GetGenes().size());

    ASSERT_EQ(genotypes_old.size(), genotypes_new.size());

    for (uint i = 0; i < genotypes_old.size(); i++) {
        ASSERT_EQ(genotypes_old.at(i).species_id, genotypes_new.at(i).species_id);
        ASSERT_EQ(genotypes_old.at(i).genes.size(), genotypes_new.at(i).genes.size());
        ASSERT_EQ(genotypes_old.at(i).nodes.size(), genotypes_new.at(i).nodes.size());
    }
}

TEST(neat_algorithms, Mutate_AddNode) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes_new, genotypes_old;
    GenePool gene_pool_new, gene_pool_old;
    NEAT neat;
    const uint n_sensors = 2, n_output = 1, n_genotypes = 6;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    gene_pool_new = neat.GetGenePool();
    gene_pool_old = neat.GetGenePool();
    genotypes_old = neat.GetGenotypes();
    genotypes_new = neat.GetGenotypes();

    Mutate(genotypes_new, gene_pool_new, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, true, true);

    ASSERT_EQ(gene_pool_old.GetNSensorNodes(), gene_pool_new.GetNSensorNodes());
    ASSERT_EQ(gene_pool_old.GetNOutputNodes(), gene_pool_new.GetNOutputNodes());
    ASSERT_EQ(gene_pool_old.GetNHiddenNodes() + n_genotypes, gene_pool_new.GetNHiddenNodes());
    ASSERT_EQ(gene_pool_old.GetNTotalNodes() + n_genotypes, gene_pool_new.GetNTotalNodes());
    ASSERT_EQ(gene_pool_old.GetGenes().size() + 2 * n_genotypes, gene_pool_new.GetGenes().size());

    ASSERT_EQ(genotypes_old.size(), genotypes_new.size());

    for (uint i = 0; i < genotypes_old.size(); i++) {
        ASSERT_EQ(genotypes_old.at(i).species_id, genotypes_new.at(i).species_id);
        ASSERT_EQ(genotypes_old.at(i).genes.size() + 2, genotypes_new.at(i).genes.size());
        ASSERT_EQ(genotypes_old.at(i).nodes.size() + 1, genotypes_new.at(i).nodes.size());
    }
}

TEST(neat_algorithms, AdjustStagnationControll) {
    const std::vector<double> fintesses{1.0, 2.0, 3.0};
    double best_fitness1(1.0), best_fitness2(3.1);
    uint counter(0);

    AdjustStagnationControll(fintesses, best_fitness1, counter);

    ASSERT_EQ(0, counter);
    ASSERT_DOUBLE_EQ(3.0, best_fitness1);

    AdjustStagnationControll(fintesses, best_fitness2, counter);

    ASSERT_EQ(1, counter);
    ASSERT_DOUBLE_EQ(3.1, best_fitness2);
}

TEST(neat_algorithms, RepopulateWithBestSpecies) {
    NEAT::Config config;
    std::vector<genome::Genotype> genotypes_new, genotypes_old;
    GenePool gene_pool_new, gene_pool_old;
    NEAT neat;
    std::vector<genome::Genotype> genotypes;
    std::vector<genome::Species> species;
    std::vector<double> fitnesses(10, 0.0);
    const uint n_sensors = 2, n_output = 1, n_genotypes = 10;
    const double weight(-123.45);
    const uint modified_genotype(5);

    config.n_sprared_genotypes = 1;

    neat.Initialize(n_sensors, n_output, n_genotypes, config);

    genotypes = neat.GetGenotypes();

    for (auto& gene : genotypes.at(modified_genotype).genes) {
        gene.enabled = false;
        gene.weight = weight;
    }

    fitnesses.at(modified_genotype) = 1000.0;

    RepopulateWithBestSpecies(fitnesses, genotypes, species, n_genotypes, config.species_distance, config.coeff1,
                              config.coeff2, config.coeff3, 1);

    ASSERT_EQ(1, species.size());
    ASSERT_EQ(n_genotypes, genotypes.size());

    for (auto genotype : genotypes) {
        ASSERT_EQ(0, genotype.species_id);

        for (auto& gene : genotype.genes) {
            ASSERT_DOUBLE_EQ(weight, gene.weight);
            ASSERT_FALSE(gene.enabled);
        }
    }
}

}  // namespace neat_algorithms
}  // namespace neat
