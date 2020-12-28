#include "mutator.h"

namespace neat {

Mutator::Mutator() {}

void Mutator::SetConfig(const Config& config) { config_ = config; }

void Mutator::SetrRandom(std::shared_ptr<Random> random) { random_ = random; }

void Mutator::Mutate(std::vector<Genome>& genomes, uint& innovation) {
    last_genes_.clear();

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
            innovation = AdjustAddNodeGenes(genome, last_genes_, gene_id, innovation, last_innov);
        }
    } else if (rand < config_.prob_new_connection) {
        const uint in(SelectId(genome.nodes_, *random_));
        const uint out(SelectId(genome.nodes_, *random_));
        const uint last_innov(innovation);

        innovation = genome.AddConnection(in, out, innovation, config_.allow_self_connection,
                                          config_.allow_recurring_connection);

        if (last_innov != innovation) {
            AdjustAddConnectionGenes(genome, last_genes_, in, out, innovation, last_innov);
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
