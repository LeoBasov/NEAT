#include "genome.h"

namespace neat {

Genome::Genome() {}

Genome::Genome(uint n_sensor_nodes, uint n_output_nodes) { Initialize(n_sensor_nodes, n_output_nodes); }

void Genome::Clear() {
    n_sensor_nodes_ = 0;
    n_output_nodes_ = 0;
    n_hidden_nodes_ = 0;
    species_id_ = 0;
    nodes_.clear();
    genes_.clear();
}

void Genome::Initialize(const uint n_sensor_nodes, const uint n_output_nodes) {
    uint innov(0);

    Clear();

    n_sensor_nodes_ = n_sensor_nodes;
    n_output_nodes_ = n_output_nodes;

    // initialize nodes
    for (uint i = 0; i < (n_sensor_nodes_ + n_output_nodes_ + 1); i++) {
        nodes_.push_back(i);
    }

    // initialize genes
    for (unsigned int i = 0; i < (n_sensor_nodes_ + 1); i++) {
        for (unsigned int o = (1 + n_sensor_nodes_); o < nodes_.size(); o++) {
            genes_.push_back(Gene(i, o, innov++));
        }
    }
}

uint Genome::AddNode(const uint gene_id, uint innov) {
    n_hidden_nodes_++;

    nodes_.push_back(nodes_.back() + 1);

    genes_.at(gene_id).enabled = false;

    genes_.push_back(Gene(genes_.at(gene_id).in, nodes_.back(), ++innov));
    genes_.push_back(Gene(nodes_.back(), genes_.at(gene_id).out, ++innov, genes_.at(gene_id).weight));

    return innov;
}

uint Genome::AddConnection(const uint in, const uint out, uint innov, const bool allow_self_connection,
                           const bool allow_recurring_connection) {
    if (out >= nodes_.size()) {
        throw std::domain_error("out_node out of bounds");
    } else if (in >= nodes_.size()) {
        throw std::domain_error("in_node out of bounds");
    }

    // sensor nodes can not be out nodes of new connection
    if (out <= n_sensor_nodes_) {
        return innov;
    } else if (!allow_self_connection && (in == out)) {
        return innov;
    }

    for (size_t i = 0; i < genes_.size(); i++) {
        if (genes_.at(i).in == in && genes_.at(i).out == out) {
            return innov;
        } else if (!allow_recurring_connection && (genes_.at(i).in == out && genes_.at(i).out == in)) {
            return innov;
        }
    }

    genes_.push_back(Gene(in, out, ++innov));

    return innov;
}

double Genome::Distance(const Genome& genome1, const Genome& genome2, const std::array<double, 3>& coefficient) {
    const uint size(std::max(genome1.genes_.size(), genome2.genes_.size()));
    double average_weight(0.0);
    uint n_disjoint(0), n_excess(0), n_common(0);
    uint i = 0, j = 0;

    for (; (i < genome1.genes_.size()) && (j < genome2.genes_.size());) {
        if (genome1.genes_.at(i).innov < genome2.genes_.at(j).innov) {
            n_disjoint++;
            i++;
        } else if (genome1.genes_.at(i).innov > genome2.genes_.at(j).innov) {
            n_disjoint++;
            j++;
        } else {
            average_weight += std::abs(genome1.genes_.at(i).weight - genome2.genes_.at(j).weight);
            n_common++;
            i++;
            j++;
        }
    }

    if (genome1.genes_.size() > i) {
        n_excess = genome1.genes_.size() - i;
    } else if (genome2.genes_.size() > j) {
        n_excess = genome2.genes_.size() - j;
    }

    if (n_common) {
        average_weight /= static_cast<double>(n_common);
    } else {
        average_weight = 0.0;
    }

    return coefficient.at(0) * (static_cast<double>(n_excess) / size) +
           coefficient.at(1) * (static_cast<double>(n_disjoint) / size) + coefficient.at(2) * average_weight;
}

double Genome::Distance(const Genome& other, const std::array<double, 3>& coefficient) const {
    return Distance(*this, other, coefficient);
}

void Genome::AdjustNodes(const uint n_sensor_nodes, const uint n_output_nodes) {
    n_sensor_nodes_ = n_sensor_nodes;
    n_output_nodes_ = n_output_nodes;
    n_hidden_nodes_ = 0;

    nodes_.clear();

    for (uint i = 0; i < (n_sensor_nodes_ + n_output_nodes_ + 1); i++) {
        nodes_.push_back(i);
    }

    for (auto& gene : genes_) {
        if (std::find(nodes_.begin(), nodes_.end(), gene.in) == nodes_.end()) {
            nodes_.push_back(gene.in);
            n_hidden_nodes_++;
        }

        if (std::find(nodes_.begin(), nodes_.end(), gene.out) == nodes_.end()) {
            nodes_.push_back(gene.out);
            n_hidden_nodes_++;
        }
    }

    std::sort(nodes_.begin(), nodes_.end());
}

Genome Genome::Mate(const Genome& fitter_parent, const Genome& parent, Random& random) {
    Genome child;

    child.genes_ = fitter_parent.genes_;

    for (uint i = 0, p = 0; i < child.genes_.size(); i++) {
        while ((parent.genes_.at(p).innov < child.genes_.at(i).innov) && (p < (parent.genes_.size() - 1))) {
            p++;
        }

        if (child.genes_.at(i).innov > parent.genes_.back().innov) {
            break;
        } else if ((parent.genes_.at(p).innov == child.genes_.at(i).innov) && (random.RandomNumber() < 0.5)) {
            child.genes_.at(i) = parent.genes_.at(p);
        }
    }

    child.AdjustNodes(fitter_parent.n_sensor_nodes_, fitter_parent.n_output_nodes_);

    return child;
}

std::map<size_t, size_t> Genome::GetNodePermuationMap() const {
    std::map<size_t, size_t> permutaion_map;

    for (uint i = 0; i < nodes_.size(); i++) {
        if (!permutaion_map.insert({nodes_.at(i), i}).second) {
            throw std::domain_error("insertion in permutaion map not possible");
        }
    }

    return permutaion_map;
}

}  // namespace neat
