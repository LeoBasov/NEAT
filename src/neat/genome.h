#pragma once

#include <vector>

namespace neat {
namespace genome {

struct Gene {
    Gene(unsigned int id, double weight = 1.0) : id(id), weight(weight) {}

    unsigned int id;  // id of gene in GenePool
    double weight = 1.0;
    bool enabled = true;

    inline bool operator==(const Gene& other) { return id == other.id; }
    inline bool operator!=(const Gene& other) { return !(*this == other); }

    inline bool operator<(const Gene& other) { return (*this).id < other.id; }
    inline bool operator>(const Gene& other) { return (*this).id > other.id; }
    inline bool operator<=(const Gene& other) { return !(*this > other); }
    inline bool operator>=(const Gene& other) { return !(*this < other); }
};

struct Genotype {
    std::vector<unsigned int > nodes;
    std::vector<Gene> genes;
    unsigned int species_id = 0;

    inline bool operator<(const Genotype& other) { return (*this).species_id < other.species_id; }
};

struct Species {
    Genotype ref_genotype;
    unsigned int n_member = 0;
    double total_fitness = 0.0;
};

}  // namespace genome
}  // namespace neat
