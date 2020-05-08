#pragma once

#include <vector>

namespace NEAT {
class GenePool {
   public:
    using uint = unsigned int;

    struct Node {
        enum Type { SENSOR, HIDDEN, OUTPUT, NONE };

        uint level = 0;
        Type type = NONE;
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
    NodeGroup hidden_nodes_;
    NodeGroup output_nodes_;
    std::vector<Node> nodes_;
    std::vector<Gene> genes_;
};
}  // namespace NEAT
