#include "neat_algorithms.h"

namespace neat {
namespace neat_algorithms {

bool AddNode(genome::Genotype& genotype, GenePool& pool, const uint& gene_id, const double& weight) {
    for (auto& gene : genotype.genes) {
        if (gene.id == gene_id) {
            std::pair<unsigned int, unsigned int> retval = pool.AddNode(gene_id);

            gene.enabled = false;

            genotype.nodes.push_back(pool.GetGene(retval.first).out_node);
            genotype.genes.push_back(genome::Gene(retval.first, 1.0));
            genotype.genes.push_back(genome::Gene(retval.second, weight));

            std::sort(genotype.nodes.begin(), genotype.nodes.end());
            std::sort(genotype.genes.begin(), genotype.genes.end());

            return true;
        }
    }

    return false;
}

bool AddConnection(genome::Genotype& genotype, GenePool& pool, const uint& in_node, const uint& out_node,
                   const double& weight) {
    std::pair<bool, unsigned int> retval = pool.AddConnection(in_node, out_node);

    if (retval.first) {
        if (std::find(genotype.genes.begin(), genotype.genes.end(), retval.second) != genotype.genes.end()) {
            return false;
        } else {
            genotype.genes.push_back(genome::Gene(retval.second, weight));
            std::sort(genotype.genes.begin(), genotype.genes.end());
        }
    }

    return retval.first;
}

genome::Genotype Mate(const genome::Genotype& fitter_parent, const genome::Genotype& parent, Random& random) {
    genome::Genotype child;

    // Get nodes
    child.nodes = fitter_parent.nodes;
    child.genes = fitter_parent.genes;

    // Get Genes
    for (uint i = 0, p = 0; i < child.genes.size(); i++) {
        if (child.genes.at(i).id > parent.genes.back().id) {
            break;
        } else if (parent.genes.at(p).id > child.genes.at(i).id) {
            continue;
        }

        while ((parent.genes.at(p).id < child.genes.at(i).id) && (p < (child.genes.size() - 1))) {
            p++;
        }

        if ((parent.genes.at(p).id == child.genes.at(i).id) && (random.RandomNumber() < 0.5)) {
            child.genes.at(i) = parent.genes.at(p);
        }
    }

    return child;
}

double CalcDistance(const std::vector<genome::Gene>& genome1, const std::vector<genome::Gene>& genome2,
                    const double& ceff1, const double& ceff2, const double& ceff3) {
    const uint size(std::max(genome1.size(), genome2.size()));
    double average_weight(0.0);
    uint n_disjoint(0), n_excess(0), n_common(0);
    uint i = 0, j = 0;

    for (; (i < genome1.size()) && (j < genome2.size());) {
        if (genome1.at(i).id < genome2.at(j).id) {
            n_disjoint++;
            i++;
        } else if (genome1.at(i).id > genome2.at(j).id) {
            n_disjoint++;
            j++;
        } else {
            average_weight += std::abs(genome1.at(i).weight - genome2.at(j).weight);
            n_common++;
            i++;
            j++;
        }
    }

    if ((genome1.size() - 1) > i) {
        n_excess = genome1.size() - i;
    } else if ((genome2.size() - 1) > j) {
        n_excess = genome2.size() - j;
    }

    if (n_common) {
        average_weight /= static_cast<double>(n_common);
    } else {
        average_weight = 0.0;
    }

    return ceff1 * (static_cast<double>(n_excess) / size) + ceff2 * (static_cast<double>(n_disjoint) / size) +
           ceff3 * average_weight;
}

MatrixXd Genotype2Phenotype(const genome::Genotype& genotype, const GenePool& pool) {
    MatrixXd matrix = MatrixXd::Zero(pool.GetNTotalNodes(), pool.GetNTotalNodes());

    for (const auto& gene : genotype.genes) {
        if (gene.enabled) {
            GenePool::Gene pool_gene = pool.GetGene(gene.id);

            matrix(pool_gene.out_node, pool_gene.in_node) += gene.weight;
        }
    }

    for (uint i = 0; i < 1 + pool.GetNSensorNodes(); i++) {
        matrix(i, i) = 1.0;
    }

    return matrix;
}

VectorXd SetUpNodes(const std::vector<double>& input_vaules, const GenePool& pool) {
    VectorXd vec = VectorXd::Zero(pool.GetNTotalNodes());

    vec(0) = 1.0;

    for (size_t i = 0; i < input_vaules.size(); i++) {
        vec(i + 1) = input_vaules.at(i);
    }

    return vec;
}

void ExecuteNetwork(const MatrixXd& matrix, VectorXd& nodes, const uint& n_const_nodes, const double& parameter) {
    nodes = matrix * nodes;

    for (Eigen::Index i = n_const_nodes; i < nodes.rows(); i++) {
        nodes(i) = utility::Sigmoid(nodes(i), parameter);
    }
}

void SortInSpecies(std::vector<genome::Genotype>& genotypes, std::vector<genome::Species>& species,
                   const double& max_distance, const double& ceff1, const double& ceff2, const double& ceff3) {
    for (auto& genotype : genotypes) {
        if (species.size()) {
            bool found(false);

            for (uint spec_id = 0; spec_id < species.size(); spec_id++) {
                if (CalcDistance(genotype.genes, species.at(spec_id).ref_genotype.genes, ceff1, ceff2, ceff3) <
                    max_distance) {
                    species.at(spec_id).n_memeber++;
                    genotype.species_id = spec_id;
                    found = true;
                    break;
                }
            }

            if (!found) {
                genome::Species spec;

                spec.ref_genotype = genotype;
                spec.n_memeber = 1;
                genotype.species_id = species.size();

                species.push_back(spec);
            }
        } else {
            genome::Species spec;

            spec.ref_genotype = genotype;
            spec.n_memeber = 1;
            genotype.species_id = species.size();

            species.push_back(spec);
        }
    }
}

void AdjustedFitnesses(std::vector<double>& fitnesses, std::vector<genome::Species>& species,
                       const std::vector<genome::Genotype>& genotypes) {
    if (fitnesses.size() != genotypes.size()) {
        throw std::domain_error("fitness size != genotype size");
    }

    for (auto& spec : species) {
        spec.total_fitness = 0.0;
    }

    for (uint i = 0; i < fitnesses.size(); i++) {
        fitnesses.at(i) /= static_cast<double>(species.at(genotypes.at(i).species_id).n_memeber);
        species.at(genotypes.at(i).species_id).total_fitness += fitnesses.at(i);
    }
}

void SortByFitness(const std::vector<double>& fitnesses, std::vector<genome::Genotype>& genotypes) {
    auto permutation_vector =
        utility::SortPermutation(fitnesses, [](double const& a, double const& b) { return a > b; });
    utility::ApplyPermutationInPlace(genotypes, permutation_vector);
}

void SortBySpecies(std::vector<genome::Genotype>& genotypes) {
    auto permutation_vector = utility::SortPermutation(
        genotypes, [](genome::Genotype const& a, genome::Genotype const& b) { return a.species_id < b.species_id; });
    utility::ApplyPermutationInPlace(genotypes, permutation_vector);
}

void ReproduceSpecies(const genome::Species& species, const std::vector<genome::Genotype>& genotypes,
                      std::vector<genome::Genotype>& new_genotypes, const double& n_new_genotypes) {}

void Reproduce(const std::vector<double>& fitnesses, const std::vector<genome::Species>& species,
               std::vector<genome::Genotype>& genotypes, const double& n_genotypes) {
    double total_fitness(0.0);
    std::vector<genome::Genotype> new_genotypes;

    SortByFitness(fitnesses, genotypes);
    SortBySpecies(genotypes);

    for (auto spec : species) {
        total_fitness += spec.total_fitness;
    }

    for (const auto& spec : species) {
        uint n_genotypes_loc(n_genotypes * (spec.total_fitness / total_fitness));

        ReproduceSpecies(spec, genotypes, new_genotypes, n_genotypes_loc);
    }
}

}  // namespace neat_algorithms
}  // namespace neat
