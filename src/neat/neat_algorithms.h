#pragma once

#include <Eigen/Dense>
#include <algorithm>

#include "../common/random.h"
#include "../common/utility.h"
#include "gene_pool.h"
#include "genome.h"

namespace neat {
namespace neat_algorithms {

bool AddNode(genome::Genotype& genotype, GenePool& pool, const uint& gene_id, const double& weight);
bool AddConnection(genome::Genotype& genotype, GenePool& pool, const uint& in_node, const uint& out_node,
                   const double& weight);
genome::Genotype Mate(const genome::Genotype& fitter_parent, const genome::Genotype& parent, Random& random);
double CalcDistance(const std::vector<genome::Gene>& genome1, const std::vector<genome::Gene>& genome2,
                    const double& ceff1, const double& ceff2, const double& ceff3);
MatrixXd Genotype2Phenotype(const genome::Genotype& genotype, const GenePool& pool);
VectorXd SetUpNodes(const std::vector<double>& input_vaules, const GenePool& pool);

// Const nodes must include bias node
void ExecuteNetwork(const MatrixXd& matrix, VectorXd& nodes, const uint& n_const_nodes, const double& parameter = 1.0);
void SortInSpecies(std::vector<genome::Genotype>& genotypes, std::vector<genome::Species>& species,
                   const double& max_distance, const double& ceff1, const double& ceff2, const double& ceff3);
void AdjustedFitnesses(std::vector<double>& fitnesses, std::vector<genome::Species>& species,
                       const std::vector<genome::Genotype>& genotypes);
void SortByFitness(const std::vector<double>& fitnesses, std::vector<genome::Genotype>& genotypes);
void SortBySpecies(std::vector<genome::Genotype>& genotypes);
void ReproduceSpecies(const genome::Species& species, const std::vector<genome::Genotype>& genotypes,
                      std::vector<genome::Genotype>& new_genotypes, const uint& n_new_genotypes,
                      const uint& species_id);
void Reproduce(const std::vector<double>& fitnesses, const std::vector<genome::Species>& species,
               std::vector<genome::Genotype>& genotypes, const uint& n_genotypes);

}  // namespace neat_algorithms
}  // namespace neat
