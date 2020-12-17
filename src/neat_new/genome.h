#pragma once

#include <vector>

namespace neat {
using uint = unsigned int;
class Genome {
   public:
    struct Gene {
        Gene() {}
        Gene(uint in, uint out, uint innov, double weight = 1.0) : in(in), out(out), innov(innov), weight(weight) {}

        uint in = 0, out = 0, innov = 0;
        double weight = 1.0;
        bool enabled = true;
    };

   public:
    Genome();
    Genome(uint n_sensor_nodes, uint n_output_nodes);
    ~Genome() = default;

    void Clear();
    void Initialize(uint n_sensor_nodes, uint n_output_nodes);

    void AddNode(uint gene_id, uint innov);

    // The fist node is allways the bias node, followd by sensor nodes, followed by output nodes.
    // All hidden nodes follow after.
    // n_sensor_nodes_ does NOT include the bias node.
    // Nodes are allways asumed to be sorted from smallest to biggest. The same for genes regading their innov number.
    uint n_sensor_nodes_ = 0, n_output_nodes_ = 0, n_hidden_nodes_ = 0;
    std::vector<uint> nodes_;
    std::vector<Gene> genes_;
};
}  // namespace neat
