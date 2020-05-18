#include "phenotype.h"

namespace NEAT {

Phenotype::Phenotype() {}

void Phenotype::AddGene(const Gene& gene) {
    genes_.push_back(gene);
    std::sort(genes_.begin(), genes_.end());
}
void Phenotype::AddGene(const uint& gene_id) {
    Gene gene;

    gene.id = gene_id;
    AddGene(gene);
}

bool Phenotype::AddGeneWithCheck(const uint& gene_id) {
    for (auto gene_loc : genes_) {
        if (gene_loc.id == gene_id) {
            return false;
        }
    }

    AddGene(gene_id);
    return true;
}

}  // namespace NEAT
