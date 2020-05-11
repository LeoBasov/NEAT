#include "phenotype.h"

namespace NEAT {

Phenotype::Phenotype() {}

void Phenotype::AddGene(const Gene& gene) { genes_.push_back(gene); }

}  // namespace NEAT
