#include "phenotype.h"

namespace NEAT {

Phenotype::Phenotype() {}

bool Phenotype::HasNode(const uint& node_id) const {
    for (uint i = 0; i < node_ids_.size(); i++) {
        if (node_ids_.at(i) == node_id) {
            return true;
        }
    }

    return false;
}

void Phenotype::AddGene(const Gene& gene, const uint& node_in, const uint& node_out) {
    genes_.push_back(gene);

    if(!HasNode(node_in)){
        node_ids_.push_back(node_in);
    }
    if(!HasNode(node_out)){
        node_ids_.push_back(node_out);
    }

    std::sort(node_ids_.begin(), node_ids_.end());
}

}  // namespace NEAT
