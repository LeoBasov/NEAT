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
                   const double& weight, const bool& allow_self_connection, const bool& allow_recurring_connection) {
    std::pair<bool, unsigned int> retval =
        pool.AddConnection(in_node, out_node, allow_self_connection, allow_recurring_connection);

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

    if (genome1.size() > i) {
        n_excess = genome1.size() - i;
    } else if (genome2.size() > j) {
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
    MatrixXd matrix = MatrixXd::Zero(genotype.nodes.size(), genotype.nodes.size());
    const std::map<uint, uint> permutation_map(neat_algorithms::GetPermutationMap(genotype));

    for (const auto& gene : genotype.genes) {
        if (gene.enabled) {
            const GenePool::Gene pool_gene = pool.GetGene(gene.id);
            const uint in_node(permutation_map.at(pool_gene.in_node));
            const uint out_node(permutation_map.at(pool_gene.out_node));

            matrix(out_node, in_node) += gene.weight;
        }
    }

    for (uint i = 0; i < 1 + pool.GetNSensorNodes(); i++) {
        matrix(i, i) = 1.0;
    }

    return matrix;
}

std::map<uint, uint> GetPermutationMap(const genome::Genotype& genotype) {
    std::map<uint, uint> permutaion_map;

    for (uint i = 0; i < genotype.nodes.size(); i++) {
        permutaion_map.insert({genotype.nodes.at(i), i});
    }

    return permutaion_map;
}

VectorXd SetUpNodes(const std::vector<double>& input_vaules, const uint& node_size) {
    VectorXd vec = VectorXd::Zero(node_size);

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
    std::vector<bool> sorted(species.size(), false);

    for (auto& spec : species) {
        spec.n_member = 0;
    }

    for (auto& genotype : genotypes) {
        if (species.size()) {
            bool found(false);

            for (uint spec_id = 0; spec_id < species.size(); spec_id++) {
                if (CalcDistance(genotype.genes, species.at(spec_id).ref_genotype.genes, ceff1, ceff2, ceff3) <
                    max_distance) {
                    species.at(spec_id).n_member++;
                    genotype.species_id = spec_id;
                    found = true;

                    if (spec_id < sorted.size() && !sorted.at(spec_id)) {
                        species.at(spec_id).ref_genotype = genotype;
                        sorted.at(spec_id) = true;
                    }

                    break;
                }
            }

            if (!found) {
                genome::Species spec;

                spec.ref_genotype = genotype;
                spec.n_member = 1;
                genotype.species_id = species.size();

                species.push_back(spec);
            }
        } else {
            genome::Species spec;

            spec.ref_genotype = genotype;
            spec.n_member = 1;
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
        fitnesses.at(i) /= static_cast<double>(species.at(genotypes.at(i).species_id).n_member);
        species.at(genotypes.at(i).species_id).total_fitness += fitnesses.at(i);
    }
}

void SortByFitness(const std::vector<double>& fitnesses, std::vector<genome::Genotype>& genotypes) {
    std::vector<std::pair<double, uint>> sorted(fitnesses.size());
    uint spec_id(0), old_id(0);

    for (uint i = 0; i < fitnesses.size(); i++) {
        sorted.at(i).first = fitnesses.at(i);
        sorted.at(i).second = i;
    }

    for (uint i = 0; i < fitnesses.size(); i++) {
        if (genotypes.at(i).species_id != spec_id) {
            std::sort(sorted.begin() + old_id, sorted.begin() + i, utility::greater());

            //--------------------------------------------------------------------
            std::vector<bool> done(genotypes.size());
            for (std::size_t k = old_id; k < i; ++k) {
                if (done[k]) {
                    continue;
                }
                done[k] = true;
                std::size_t prev_j = k;
                std::size_t j = sorted[k].second;
                while (k != j) {
                    std::swap(genotypes[prev_j], genotypes[j]);
                    done[j] = true;
                    prev_j = j;
                    j = sorted[j].second;
                }
            }
            //---------------------------------------------------------------------

            spec_id = genotypes.at(i).species_id;
            old_id = i;
        } else if (i == fitnesses.size() - 1) {
            std::sort(sorted.begin() + old_id, sorted.begin() + i + 1, utility::greater());

            //--------------------------------------------------------------------
            std::vector<bool> done(genotypes.size());
            for (std::size_t k = old_id; k < i; ++k) {
                if (done[k]) {
                    continue;
                }
                done[k] = true;
                std::size_t prev_j = k;
                std::size_t j = sorted[k].second;
                while (k != j) {
                    std::swap(genotypes[prev_j], genotypes[j]);
                    done[j] = true;
                    prev_j = j;
                    j = sorted[j].second;
                }
            }
            //---------------------------------------------------------------------
        }
    }
}

void SortBySpecies(std::vector<genome::Genotype>& genotypes) {
    auto permutation_vector = utility::SortPermutation(
        genotypes, [](genome::Genotype const& a, genome::Genotype const& b) { return a.species_id < b.species_id; });
    utility::ApplyPermutationInPlace(genotypes, permutation_vector);
}

void ReproduceSpecies(const genome::Species& species, const std::vector<genome::Genotype>& genotypes,
                      std::vector<genome::Genotype>& new_genotypes, const uint& n_new_genotypes, const uint& species_id,
                      const double& prob_mate) {
    uint n_genotypes(0);
    Random random;

    for (uint i = 0; i < genotypes.size(); i++) {
        if (genotypes.at(i).species_id == species_id) {
            while (n_genotypes < n_new_genotypes) {
                for (uint j = i; j < i + species.n_member; j++) {
                    if (n_genotypes >= n_new_genotypes) {
                        return;
                    } else if ((random.RandomNumber() < prob_mate) && (j < i + species.n_member - 1)) {
                        new_genotypes.push_back(Mate(genotypes.at(j), genotypes.at(j + 1), random));
                        n_genotypes++;
                    } else {
                        new_genotypes.push_back(genotypes.at(j));
                        n_genotypes++;
                    }
                }
            }
            break;
        }
    }
}

void Reproduce(const std::vector<double>& fitnesses, const std::vector<genome::Species>& species,
               std::vector<genome::Genotype>& genotypes, const uint& n_genotypes, const double& prob_mate) {
    double total_fitness(0.0);
    std::vector<genome::Genotype> new_genotypes;

    SortBySpecies(genotypes);
    SortByFitness(fitnesses, genotypes);

    for (auto spec : species) {
        total_fitness += spec.total_fitness;
    }

    for (uint i = 0; i < species.size(); i++) {
        uint n_genotypes_loc(n_genotypes * (species.at(i).total_fitness / total_fitness));

        ReproduceSpecies(species.at(i), genotypes, new_genotypes, n_genotypes_loc, i, prob_mate);
    }

    genotypes = new_genotypes;
}

void ReproduceBestSpecies(const std::vector<double>& fitnesses, const std::vector<genome::Species>& species,
                          std::vector<genome::Genotype>& genotypes, const uint& n_genotypes, const double& prob_mate,
                          const uint& n_species) {
    double total_fitness(0.0);
    std::vector<genome::Genotype> new_genotypes;

    SortBySpecies(genotypes);
    SortByFitness(fitnesses, genotypes);

    for (auto spec : species) {
        total_fitness += spec.total_fitness;
    }

    auto permutation_vector = utility::SortPermutation(
        species, [](genome::Species const& a, genome::Species const& b) { return a.total_fitness > b.total_fitness; });

    for (uint i = 0; i < n_species; i++) {
        const uint species_id(permutation_vector.at(i));
        uint n_genotypes_loc(n_genotypes * (species.at(species_id).total_fitness / total_fitness));

        ReproduceSpecies(species.at(species_id), genotypes, new_genotypes, n_genotypes_loc, species_id, prob_mate);
    }

    genotypes = new_genotypes;
}

void Mutate(std::vector<genome::Genotype>& genotypes, GenePool& pool, const double& prob_weight_change,
            const double& prob_new_weight, const double& prob_new_node, const double& prob_new_connection,
            const double& weight_min, const double& weight_max, const bool& allow_self_connection,
            const bool& allow_recurring_connection) {
    Random random;

    for (auto& genotype : genotypes) {
        const double rand(random.RandomNumber());
        const uint gene_genome_id = static_cast<uint>(std::round(random.RandomNumber(0.0, genotype.genes.size() - 1)));
        const uint gene_id(genotype.genes.at(gene_genome_id).id);
        const uint node_genome_id1 = static_cast<uint>(std::round(random.RandomNumber(0.0, genotype.nodes.size() - 1)));
        const uint node_genome_id2 = static_cast<uint>(std::round(random.RandomNumber(0.0, genotype.nodes.size() - 1)));
        const uint node_id1(genotype.nodes.at(node_genome_id1));
        const uint node_id2(genotype.nodes.at(node_genome_id2));

        if (rand < prob_new_node) {
            AddNode(genotype, pool, gene_id, genotype.genes.at(gene_genome_id).weight);
        } else if (rand < prob_new_connection) {
            AddConnection(genotype, pool, node_id1, node_id2, random.RandomNumber(weight_min, weight_max),
                          allow_self_connection, allow_recurring_connection);
        } else if (rand < prob_weight_change) {
            if (random.RandomNumber() < prob_new_weight) {
                AssignNewWeight(genotype, gene_genome_id, weight_min, weight_max);
            } else {
                const double perturbation_fraq(0.1);
                PertubateWeight(genotype, gene_genome_id, perturbation_fraq);
            }
        }
    }
}

void AssignNewWeight(genome::Genotype& genotype, const uint& gene_genome_id, const double& weight_min,
                     const double& weight_max) {
    Random random;

    genotype.genes.at(gene_genome_id).weight = random.RandomNumber(weight_min, weight_max);
}

void PertubateWeight(genome::Genotype& genotype, const uint& gene_genome_id, const double& perturbation_fraq) {
    Random random;

    if (genotype.genes.at(gene_genome_id).weight >= 0.0) {
        genotype.genes.at(gene_genome_id).weight +=
            random.RandomNumber(-perturbation_fraq * genotype.genes.at(gene_genome_id).weight,
                                perturbation_fraq * genotype.genes.at(gene_genome_id).weight);
    } else {
        genotype.genes.at(gene_genome_id).weight +=
            random.RandomNumber(perturbation_fraq * genotype.genes.at(gene_genome_id).weight,
                                -perturbation_fraq * genotype.genes.at(gene_genome_id).weight);
    }
    // random.NormalRandomNumber(genotype.genes.at(gene_genome_id).weight, 1.0);
}

void AdjustStagnationControll(const std::vector<double>& fitnesses, double& best_fitness, uint& unimproved_counter) {
    const double best_fitness_old(best_fitness);

    for (auto fitness : fitnesses) {
        if (fitness > best_fitness) {
            best_fitness = fitness;
        }
    }

    if (!(best_fitness > best_fitness_old)) {
        unimproved_counter++;
    }
}

void RepopulateWithBestSpecies(std::vector<double>& fitnesses, std::vector<genome::Genotype>& genotypes,
                               std::vector<genome::Species>& species, const uint& n_genotypes_init,
                               const double& species_distance, const double& coeff1, const double& coeff2,
                               const double& coeff3, const uint& n_sprared_genotypes) {
    auto permutation_vector =
        utility::SortPermutation(fitnesses, [](const double& a, const double& b) { return a > b; });
    utility::ApplyPermutationInPlace(genotypes, permutation_vector);
    utility::ApplyPermutationInPlace(fitnesses, permutation_vector);

    genotypes.resize(n_genotypes_init);

    for (uint i = n_sprared_genotypes - 1, j = 0; i < n_genotypes_init; i++) {
        if (j >= n_sprared_genotypes - 1) {
            j = 0;
        }

        genotypes.at(i) = genotypes.at(j++);
    }

    std::random_shuffle(genotypes.begin(), genotypes.end());
    fitnesses = std::vector<double>(genotypes.size(), 1.0);

    neat_algorithms::SortInSpecies(genotypes, species, species_distance, coeff1, coeff2, coeff3);
}

}  // namespace neat_algorithms
}  // namespace neat
