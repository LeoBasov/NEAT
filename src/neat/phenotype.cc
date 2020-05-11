#include "phenotype.h"

namespace NEAT {

Phenotype::Phenotype() {}

void Phenotype::AddGene(const Gene& gene) { genes_.push_back(gene); }
void Phenotype::AddGene(const uint& gene_id) {
    Gene gene;

    gene.id = gene_id;
    AddGene(gene);
}

}  // namespace NEAT
