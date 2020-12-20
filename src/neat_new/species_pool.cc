#include "species_pool.h"

namespace neat {

SpeciesPool::SpeciesPool() {}

void SpeciesPool::Clear() {
    species_.clear();
    total_fitness_ = 0.0;
}

void SpeciesPool::SetConfig(const Config& config) { config_ = config; }

std::vector<SpeciesPool::Species> SpeciesPool::GetSpecies() const { return species_; }

double SpeciesPool::GetTotalFitness() const { return total_fitness_; }

void SpeciesPool::SortInSpecies(std::vector<Genome>& genomes) {
    for (auto& spec : species_) {
        spec.n_member = 0;
        spec.set_ref_genome = false;
    }

    if (!species_.size()) {
        Species spec;

        spec.ref_genome = genomes.front();
        spec.n_member = 0;
        genomes.front().species_id_ = 0;

        species_.push_back(spec);
    }

    for (auto& genome : genomes) {
        bool found(false);

        for (uint spec_id = 0; spec_id < species_.size(); spec_id++) {
            const double distance(species_.at(spec_id).ref_genome.Distance(genome, config_.distance_coefficients));

            if (distance < config_.max_species_distance) {
                species_.at(spec_id).n_member++;
                species_.at(spec_id).SetRefGenome(genome);
                genome.species_id_ = spec_id;
                found = true;

                break;
            }
        }

        if (!found) {
            Species spec;

            spec.ref_genome = genome;
            spec.n_member = 1;
            genome.species_id_ = species_.size();

            species_.push_back(spec);
        }
    }
}

void SpeciesPool::AdjustFitnesses(std::vector<double>& fitnesses, const std::vector<Genome>& genomes) {
    if (fitnesses.size() != genomes.size()) {
        throw std::domain_error("fitness size != genomes size");
    }

    for (auto& spec : species_) {
        spec.total_fitness = 0.0;
    }

    total_fitness_ = 0.0;

    for (uint i = 0; i < fitnesses.size(); i++) {
        if (!species_.at(genomes.at(i).species_id_).n_member) {
            throw std::domain_error("empty species in AdjustFitnesses");
        }

        fitnesses.at(i) /= static_cast<double>(species_.at(genomes.at(i).species_id_).n_member);
        species_.at(genomes.at(i).species_id_).total_fitness += fitnesses.at(i);
        total_fitness_ += fitnesses.at(i);
    }
}

void SpeciesPool::Reproduce(std::vector<Genome>& genotypes, std::vector<double>& fitnesses, const uint& n_genotypes) {
    std::vector<Genome> new_genotypes;

    SortBySpecies(fitnesses, genotypes);
    SortByFitness(fitnesses, genotypes);

    for (uint i = 0; i < species_.size(); i++) {
        uint n_genotypes_loc(n_genotypes * (species_.at(i).total_fitness / total_fitness_));

        ReproduceSpecies(species_.at(i), genotypes, new_genotypes, n_genotypes_loc, i, config_.prob_mate);
    }

    genotypes = new_genotypes;
}

void SpeciesPool::ReproduceSpecies(const Species& species, const std::vector<Genome>& genotypes,
                                   std::vector<Genome>& new_genotypes, const uint& n_new_genotypes,
                                   const uint& species_id, const double& prob_mate) {
    if (!species.n_member) {
        return;
    }

    // const uint n_repoducable(species.n_member > 3 ? species.n_member / 3 : 1);
    const uint n_repoducable(species.n_member);
    uint n_genotypes(0);
    Random random;

    for (uint i = 0; i < genotypes.size(); i++) {
        if (genotypes.at(i).species_id_ == species_id) {
            while (n_genotypes < n_new_genotypes) {
                for (uint j = i; j < i + n_repoducable; j++) {
                    // const double probability(1.0);
                    // const double probability(static_cast<double>(n_repoducable + i - j)/n_repoducable);

                    const double factor(-1.0 / (n_repoducable * n_repoducable));
                    const double probability(factor * (j - i) * (j - i) + 1);

                    if (n_genotypes > n_new_genotypes) {
                        return;
                    } else if (random.RandomNumber() > probability) {
                        continue;
                    } else if ((random.RandomNumber() < prob_mate) && (j < i + species.n_member - 1)) {
                        new_genotypes.push_back(Genome::Mate(genotypes.at(j), genotypes.at(j + 1), random));
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

void SpeciesPool::SortBySpecies(std::vector<double>& fitnesses, std::vector<Genome>& genotypes) {
    auto permutation_vector = utility::SortPermutation(
        genotypes, [](Genome const& a, Genome const& b) { return a.species_id_ < b.species_id_; });
    utility::ApplyPermutationInPlace(genotypes, permutation_vector);
    utility::ApplyPermutationInPlace(fitnesses, permutation_vector);
}

void SpeciesPool::SortByFitness(std::vector<double>& fitnesses, std::vector<Genome>& genotypes) {
    std::vector<std::pair<double, uint>> sorted(fitnesses.size());
    uint spec_id(0), old_id(0);

    for (uint i = 0; i < fitnesses.size(); i++) {
        sorted.at(i).first = fitnesses.at(i);
        sorted.at(i).second = i;
    }

    for (uint i = 0; i < fitnesses.size(); i++) {
        if (genotypes.at(i).species_id_ != spec_id) {
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
                    std::swap(fitnesses[prev_j], fitnesses[j]);
                    done[j] = true;
                    prev_j = j;
                    j = sorted[j].second;
                }
            }
            //---------------------------------------------------------------------

            spec_id = genotypes.at(i).species_id_;
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
                    std::swap(fitnesses[prev_j], fitnesses[j]);
                    done[j] = true;
                    prev_j = j;
                    j = sorted[j].second;
                }
            }
            //---------------------------------------------------------------------
        }
    }
}

}  // namespace neat
