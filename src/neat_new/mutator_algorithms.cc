#include "mutator_algorithms.h"

namespace neat {
namespace mutator_algorithms {

void PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq) {
    genome.genes_.at(gene_id).weight *= (1.0 - perturbation_fraq) + 2 * (perturbation_fraq)*random.RandomNumber();
}

double RandomizeWeight(const double& min, const double& max, Random& random) {
    return min + (max - min) * random.RandomNumber();
}

std::pair<bool, uint> InLastGenes(const uint& in, const uint& out, const std::vector<LastGene>& last_genes,
                                  LastGene::Type type) {
    for (uint i = 0; i < last_genes.size(); i++) {
        if ((in == last_genes.at(i).in) && (out == last_genes.at(i).out) && (last_genes.at(i).type == type)) {
            return {true, i};
        }
    }

    return {false, 0};
}

uint AdjustAddNodeGenes(Genome& genome, std::vector<LastGene>& last_genes, const uint& gene_id, const uint& innovation,
                        const uint& last_innovation) {
    const uint in(genome.genes_.at(gene_id).in), out(genome.genes_.at(gene_id).out);
    const std::pair<bool, uint> ret_pair(InLastGenes(in, out, last_genes, LastGene::ADD_NODE));
    LastGene last_gene;

    if (ret_pair.first) {
        genome.genes_.at(genome.genes_.size() - 2) = last_genes.at(ret_pair.second).genes.first;
        genome.genes_.at(genome.genes_.size() - 1) = last_genes.at(ret_pair.second).genes.second;

        std::sort(genome.genes_.begin(), genome.genes_.end());
        genome.AdjustNodes(genome.n_sensor_nodes_, genome.n_output_nodes_);

        return last_innovation;
    } else {
        last_gene.type = LastGene::ADD_NODE;
        last_gene.in = in;
        last_gene.out = out;
        last_gene.genes.first = genome.genes_.at(genome.genes_.size() - 2);
        last_gene.genes.second = genome.genes_.at(genome.genes_.size() - 1);

        last_genes.push_back(last_gene);

        return innovation;
    }
}

}  // namespace mutator_algorithms
}  // namespace neat
