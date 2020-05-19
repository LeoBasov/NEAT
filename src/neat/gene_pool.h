#pragma once

#include <array>
#include <vector>

namespace NEAT {
class GenePool {
   public:
    using uint = unsigned int;

    struct Node {
        Node(const uint& level = 0) : level(level) {}

        uint level;
    };

    struct Gene {
        Gene(const uint& in = 0, const uint& out = 0) : in(in), out(out) {}

        uint in;
        uint out;
        bool conected = false;
        std::array<uint, 3> gene_ids;
    };

    struct NodeGroup {
        uint ofset = 0;
        uint n_parts = 0;

        void Clear() {
            ofset = 0;
            n_parts = 0;
        }
    };

    GenePool();
    ~GenePool() = default;

    void Clear();
    void Initialize(const uint& n_input, const uint& n_output);
    bool AddNode(const uint& node_in, const uint& node_out);
    std::pair<bool, unsigned int> AddConnection(const uint& node_in, const uint& node_out);
    void AdjustLevelsAbove(const uint level);

    NodeGroup input_nodes_;
    NodeGroup output_nodes_;
    NodeGroup hidden_nodes_;
    // All nodes are sorted in a signle list.
    // First input, then output, then everything else.
    // No input or output node can be added later.
    std::vector<Node> nodes_;
    std::vector<Gene> genes_;
};
}  // namespace NEAT
