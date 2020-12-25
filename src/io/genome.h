#pragma once

#include <fstream>

#include "../exception/exception.h"
#include "../neat_new/genome.h"

namespace neat {
namespace genome {

Genome ReadGenome(const std::string& file_name, const uint& n_sensor_nodes, const uint& n_output_nodes);
}
}  // namespace neat
