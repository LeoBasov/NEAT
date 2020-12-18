#include "mutator.h"

namespace neat {

Mutator::Mutator() {}

void Mutator::SetConfig(const Config& config) { config_ = config; }

void Mutator::SetrRandom(const Random& random) { random_ = random; }

void Mutator::Mutate(Genome& genome, uint& innovation) {
    const double rand(random_.RandomNumber());

    if (rand < config_.prob_new_node) {
        const uint gene_id(random_.RandomIntNumber(0, genome.genes_.size() - 1));

        innovation = genome.AddNode(gene_id, innovation);
    } else if (rand < config_.prob_new_connection) {
        const uint in(genome.nodes_.at(random_.RandomIntNumber(0, genome.nodes_.size() - 1)));
        const uint out(genome.nodes_.at(random_.RandomIntNumber(0, genome.nodes_.size() - 1)));

        innovation = genome.AddConnection(in, out, innovation, config_.allow_self_connection,
                                          config_.allow_recurring_connection);
    } else if (rand < config_.prob_weight_change) {
        if (random_.RandomNumber() < config_.prob_new_weight) {
            const double weight(random_.RandomNumber(config_.weight_min, config_.weight_max));
            const uint gene_id(random_.RandomIntNumber(0, genome.genes_.size() - 1));

            genome.genes_.at(gene_id).weight = weight;
        } else {
            const uint gene_id(random_.RandomIntNumber(0, genome.genes_.size() - 1));

            PertubateWeight(genome, random_, gene_id, config_.perturbation_fraction);
        }
    }
}

void Mutator::PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq) {
    genome.genes_.at(gene_id).weight *= (1.0 - perturbation_fraq) + 2 * (perturbation_fraq)*random.RandomNumber();
}

}  // namespace neat
