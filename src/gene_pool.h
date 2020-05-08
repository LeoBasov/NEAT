#pragma once

#include <vector>

namespace NEAT {
class GenePool {
   public:
    using uint = unsigned int;

    struct Node {
        uint level = 0;
    };

    struct Gene {
        uint in;
        uint out;
    };

    GenePool();
    ~GenePool() = default;

   private:
    struct NodeGroup {
        uint ofset = 0;
        uint n_parts = 0;
    };

   private:
    NodeGroup input_nodes_;
    NodeGroup output_nodes_;
    NodeGroup hidden_nodes_;
    // All noes are sorted in a signle list.
    // First input, then output, then everything else.
    // No input or output node can be added later.
    std::vector<Node> nodes_;
    std::vector<Gene> genes_;
};
}  // namespace NEAT
