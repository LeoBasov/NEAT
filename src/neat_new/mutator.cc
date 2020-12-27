#include "mutator.h"

namespace neat {

Mutator::Mutator() {}

void Mutator::SetConfig(const Config& config) { config_ = config; }

void Mutator::SetrRandom(std::shared_ptr<Random> random) { random_ = random; }

void Mutator::Mutate(std::vector<Genome>& genomes, uint& innovation) {
    // last_genes_.clear();

    for (auto& genome : genomes) {
        Mutate(genome, innovation);
    }
}

void Mutator::Mutate(Genome& genome, uint& innovation) {
    const double rand(random_->RandomNumber());

    if (rand < config_.prob_new_node) {
        const uint gene_id(SelectId(genome.genes_, *random_));
        const uint last_innov(innovation);

        innovation = genome.AddNode(gene_id, innovation);

        if (last_innov != innovation) {
            const uint in(genome.genes_.at(gene_id).in), out(genome.genes_.at(gene_id).out);
            const std::pair<bool, uint> ret_pair(InLastGenes(in, out, last_genes_, LastGene::ADD_NODE));
            LastGene last_gene;

            if (ret_pair.first) {
                genome.genes_.at(genome.genes_.size() - 2) = last_genes_.at(ret_pair.second).genes.first;
                genome.genes_.at(genome.genes_.size() - 1) = last_genes_.at(ret_pair.second).genes.second;

                std::sort(genome.genes_.begin(), genome.genes_.end());
                genome.AdjustNodes(genome.n_sensor_nodes_, genome.n_output_nodes_);

                innovation = last_innov;
            } else {
                last_gene.type = LastGene::ADD_NODE;
                last_gene.in = in;
                last_gene.out = out;
                last_gene.genes.first = genome.genes_.at(genome.genes_.size() - 2);
                last_gene.genes.second = genome.genes_.at(genome.genes_.size() - 1);

                last_genes_.push_back(last_gene);
            }
        }
    } else if (rand < config_.prob_new_connection) {
        const uint in(SelectId(genome.nodes_, *random_));
        const uint out(SelectId(genome.nodes_, *random_));
        const uint last_innov(innovation);

        innovation = genome.AddConnection(in, out, innovation, config_.allow_self_connection,
                                          config_.allow_recurring_connection);

        if (last_innov != innovation) {
            const std::pair<bool, uint> ret_pair(InLastGenes(in, out, last_genes_, LastGene::ADD_CONNECTION));
            LastGene last_gene;

            if (ret_pair.first) {
                genome.genes_.back() = last_genes_.at(ret_pair.second).genes.first;

                std::sort(genome.genes_.begin(), genome.genes_.end());

                innovation = last_innov;
            } else {
                last_gene.type = LastGene::ADD_CONNECTION;
                last_gene.in = in;
                last_gene.out = out;
                last_gene.genes.first = genome.genes_.back();

                last_genes_.push_back(last_gene);
            }
        }
    } else if (rand < config_.prob_weight_change) {
        if (random_->RandomNumber() < config_.prob_new_weight) {
            const uint gene_id(SelectId(genome.genes_, *random_));

            genome.genes_.at(gene_id).weight = RandomizeWeight(config_.weight_min, config_.weight_max, *random_);
        } else {
            const uint gene_id(SelectId(genome.genes_, *random_));

            PertubateWeight(genome, *random_, gene_id, config_.perturbation_fraction);
        }
    }
}

}  // namespace neat
