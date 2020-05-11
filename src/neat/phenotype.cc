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

}  // namespace NEAT
