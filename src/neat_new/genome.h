#pragma once

#include <vector>

namespace neat {
using uint = unsigned int;
class Genome {
   public:
    struct Gene {
        Gene() {}
        Gene(uint in, uint out, uint innov) : in(in), out(out), innov(innov) {}

        uint in, out, innov;
        double weight = 1.0;
        bool enabled = true;
    };

   public:
    Genome();
    Genome(uint n_sensor_nodes, uint n_output_nodes);
    ~Genome() = default;

    void Clear();
    uint Initialize(uint n_sensor_nodes, uint n_output_nodes);

    // The fist node is allways the bias node, followd by sensor nodes, followed by output nodes.
    // All hidden nodes follow after.
    // n_sensor_nodes_ does NOT include the bias node
    uint n_sensor_nodes_ = 0, n_output_nodes_ = 0, n_hidden_nodes_ = 0;
    std::vector<uint> nodes_;
    std::vector<Gene> genes_;
};
}  // namespace neat
