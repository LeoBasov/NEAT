#include "genome.h"

namespace neat {
namespace genome {

Genome ReadGenome(const std::string& file_name, const uint& n_sensor_nodes, const uint& n_output_nodes) {
    Genome genome;
    std::ifstream input(file_name);
    std::string line;
    uint innov(0);

    if (!input.is_open()) {
        throw Exception("Could not open file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    while (std::getline(input, line)) {
        size_t pos, lost_pos(0);
        std::vector<std::string> values;
        Genome::Gene gene;

        while ((pos = line.find(",", lost_pos)) != std::string::npos) {
            values.push_back(line.substr(lost_pos, pos - lost_pos));
            lost_pos = pos + 1;
        }

        values.push_back(line.substr(lost_pos, (line.size() - lost_pos)));

        gene.in = std::stoi(values.at(0));
        gene.out = std::stoi(values.at(1));
        gene.weight = std::stod(values.at(2));
        gene.innov = innov++;

        genome.genes_.push_back(gene);
    }

    input.close();

    if (!input.good() && !input.eof()) {
        throw Exception("Error occurred at reading file [" + file_name + "]", __PRETTY_FUNCTION__);
    }

    genome.AdjustNodes(n_sensor_nodes, n_output_nodes);

    return genome;
}

}  // namespace genome
}  // namespace neat
