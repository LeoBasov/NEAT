#include <fstream>
#include <iostream>

#include "../../src/common/timer.h"
#include "../../src/io/mnist.h"
#include "../../src/neat_new/neat.h"

using namespace neat;

std::vector<mnist::Image> GetTrainingInput(uint n_images);
std::vector<mnist::Image> GetTestInput(uint n_images);

std::vector<double> Execute(const Neat& neat, const std::vector<mnist::Image>& images);
void PrintResultsNetwork(const Neat& neat, const uint& network_id, uint n_refs);
uint FindBestNetwork(const std::vector<double>& fitnesses);
void WriteNetworkToFile(const Genome& genotype, const std::string& file_name = "best.csv");
void WriteFitnessToFile(std::ofstream& stream, std::vector<double> fitnesses, uint unimproved_counter);

int main(int, char**) {
    const uint n_iterations(1500);
    const double min_fitness(7.8);
    std::ofstream stream("fitness.csv");
    Timer exex, update, total;
    Neat::Config config;
    Neat neat;
    const uint n_sensor_nodes(784), n_output_nodes(8), n_genotypes(150);
    std::vector<double> fitnesses;
    uint best_network_id(0);
    double mean(0.0);
    const uint n_refs(5);

    std::vector<mnist::Image> input(GetTrainingInput(100));

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "START NEAT" << std::endl;

    config.species_pool_config.distance_coefficients.at(0) = 3000.0;
    config.species_pool_config.distance_coefficients.at(1) = 3000.0;
    config.species_pool_config.distance_coefficients.at(2) = 2e-3;

    config.mutator_config.allow_recurring_connection = false;
    config.mutator_config.allow_self_connection = false;

    config.mutator_config.prob_new_node = 0.003;

    config.mutator_config.weight_min = -1000.0;  // before -1000.0
    config.mutator_config.weight_max = 1000.0;   // before  1000.0

    std::cout << "INITIALIZING" << std::endl;
    neat.Initialize(n_sensor_nodes, n_output_nodes, n_genotypes, config);

    total.Start();
    for (uint i = 0; i < n_iterations; i++) {
        std::cout << "EXECUTING" << std::endl;
        exex.Start();
        fitnesses = Execute(neat, input);
        exex.Stop();

        best_network_id = FindBestNetwork(fitnesses);

        if (fitnesses.at(best_network_id) >= min_fitness) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "----- CRITERIUM ACHIEVED -----" << std::endl;
            std::cout << "BEST FITNESS: " << fitnesses.at(best_network_id) << std::endl;

            WriteNetworkToFile(neat.GetGenomes().at(best_network_id));
            WriteFitnessToFile(stream, fitnesses, 0);

            break;
        }

        std::cout << "EVOLVING" << std::endl;
        update.Start();
        neat.Evolve(fitnesses, n_genotypes);
        update.Stop();

        mean = 0.0;

        for (auto fit : fitnesses) {
            mean += fit;
        }

        mean /= fitnesses.size();
        WriteFitnessToFile(stream, fitnesses, 0);

        if (!(i % 1)) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "ITTERATION: " << i << std::endl;
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "EXECUTION TIME " << exex.GetCurrentDuration() << std::endl;
            std::cout << "UPDATE TIME " << update.GetCurrentDuration() << std::endl;
            std::cout << "N SPECIES   " << neat.GetSpeciesPool().GetSpecies().size() << std::endl;
            std::cout << "N GENOTYPES " << neat.GetGenomes().size() << std::endl;
            std::cout << "INNOVATION: " << neat.GetInnovation() << std::endl;
            std::cout << "BEST FITNESS: " << fitnesses.at(best_network_id) << std::endl;
            std::cout << "MEAN FITNESS " << mean << std::endl;
            std::cout << "------------------------------------------------------" << std::endl;
        }
    }
    total.Stop();

    PrintResultsNetwork(neat, best_network_id, n_refs);
    WriteNetworkToFile(neat.GetGenomes().at(best_network_id), "last.csv");

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "TOTAL EXECUTION TIME " << exex.GetTotalDuration() << std::endl;
    std::cout << "TOTAL UPDATE TIME " << update.GetTotalDuration() << std::endl;
    std::cout << "TOTAL TIME " << total.GetTotalDuration() << std::endl;

    std::cout << "========================================================================" << std::endl;

    return 0;
}

std::vector<mnist::Image> GetTrainingInput(uint n_images) {
    // Data set found here: http://yann.lecun.com/exdb/mnist/

    mnist::ImageHeader image_header;
    std::vector<mnist::Image> images;
    std::vector<uint> labels;
    const std::string file_name_images("/home/lbasov/AI/train-images-idx3-ubyte");
    const std::string file_name_labels("/home/lbasov/AI/train-labels-idx1-ubyte");

    // READ FILES
    image_header = mnist::ReadImageHeader(file_name_images);

    std::cout << "N IMAGES READ: " << image_header.n_images << " EXPECTED: 60000" << std::endl;
    std::cout << "N ROWS:        " << image_header.n_rows << " EXPECTED: 28" << std::endl;
    std::cout << "N COLUMNS:     " << image_header.n_columns << " EXPECTED: 28" << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " IMAGES" << std::endl;

    images = mnist::ReadImages(file_name_images, n_images);

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " LABELS" << std::endl;

    labels = mnist::ReadLabels(file_name_labels, n_images);

    for (uint i = 0; i < n_images; i++) {
        images.at(i).label = mnist::Decimal2Binray(labels.at(i));

        for (uint p = 0; p < images.at(i).pixels.size(); p++) {
            images.at(i).pixels.at(p) /= 255.0;
        }
    }

    return images;
}

std::vector<mnist::Image> GetTestInput(uint n_images) {
    // Data set found here: http://yann.lecun.com/exdb/mnist/

    mnist::ImageHeader image_header;
    std::vector<mnist::Image> images;
    std::vector<uint> labels;
    const std::string file_name_images("/home/lbasov/AI/t10k-images-idx3-ubyte");
    const std::string file_name_labels("/home/lbasov/AI/t10k-labels-idx1-ubyte");

    // READ FILES
    image_header = mnist::ReadImageHeader(file_name_images);

    std::cout << "N IMAGES READ: " << image_header.n_images << " EXPECTED: 60000" << std::endl;
    std::cout << "N ROWS:        " << image_header.n_rows << " EXPECTED: 28" << std::endl;
    std::cout << "N COLUMNS:     " << image_header.n_columns << " EXPECTED: 28" << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " IMAGES" << std::endl;

    images = mnist::ReadImages(file_name_images, n_images);

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "READING " + std::to_string(n_images) + " LABELS" << std::endl;

    labels = mnist::ReadLabels(file_name_labels, n_images);

    for (uint i = 0; i < n_images; i++) {
        images.at(i).label = mnist::Decimal2Binray(labels.at(i));

        for (uint p = 0; p < images.at(i).pixels.size(); p++) {
            images.at(i).pixels.at(p) /= 255.0;
        }
    }

    return images;
}

std::vector<double> Execute(const Neat& neat, const std::vector<mnist::Image>& images) {
    std::vector<Network> networks(neat.GetNetworks());
    std::vector<double> fitnesses(networks.size(), 0.0);

    for (uint n = 0; n < networks.size(); n++) {
        std::vector<double> loc_fitness(images.size(), 0.0);

#pragma omp parallel for
        for (uint i = 0; i < images.size(); i++) {
            std::vector<double> result = networks.at(n).Execute(images.at(i).pixels);

            if (!networks.at(n).cyclic_) {
                for (uint q = 0; q < result.size(); q++) {
                    loc_fitness.at(i) += (1.0 - std::abs(images.at(i).label.at(q) - result.at(q))) / images.size();
                }
            }
        }

        for (uint f = 0; f < loc_fitness.size(); f++) {
            fitnesses.at(n) += loc_fitness.at(f);
        }
    }

    return fitnesses;
}

void PrintResultsNetwork(const Neat& neat, const uint& network_id, uint n_refs) {
    Network network(neat.GetNetworks().at(network_id));
    std::vector<std::vector<double>> resutls;
    std::vector<mnist::Image> input(GetTestInput(n_refs));

    for (uint k = 0; k < n_refs; k++) {
        std::vector<double> result = network.Execute(input.at(k).pixels);

        std::cout << "INPUT: [";
        for (uint i = 0; i < result.size(); i++) {
            std::cout << input.at(k).label.at(i) << ",";
        }

        std::cout << "] VALUE: " + std::to_string(mnist::Binray2Decimal(input.at(k).label)) + " OUTPUT: [";

        for (uint i = 0; i < result.size(); i++) {
            std::cout << result.at(i) << ",";
        }

        std::cout << "] VALUE: " + std::to_string(mnist::Binray2Decimal(result)) << std::endl;
    }

    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "N NODES: " << neat.GetGenomes().at(network_id).nodes_.size();
    std::cout << " N GENES: " << neat.GetGenomes().at(network_id).genes_.size() << std::endl;
}

uint FindBestNetwork(const std::vector<double>& fitnesses) {
    uint id(0);
    double best(0.0);

    for (uint i = 0; i < fitnesses.size(); i++) {
        if (fitnesses.at(i) > best) {
            id = i;
            best = fitnesses.at(i);
        }
    }

    return id;
}

void WriteNetworkToFile(const Genome& genotype, const std::string& file_name) {
    std::ofstream stream(file_name);
    const std::map<size_t, size_t> permutation_map(genotype.GetNodePermuationMap());

    for (auto gene : genotype.genes_) {
        if (gene.enabled) {
            stream << permutation_map.at(gene.in) << "," << permutation_map.at(gene.out) << "," << gene.weight
                   << std::endl;
        }
    }
}

void WriteFitnessToFile(std::ofstream& stream, std::vector<double> fitnesses, uint unimproved_counter) {
    double mean(0.0), top(0.0), buttom(0.0);
    uint percetile(fitnesses.size() / 10);

    std::sort(fitnesses.rbegin(), fitnesses.rend());

    for (auto fitness : fitnesses) {
        mean += fitness;
    }

    for (uint i = 0; i < percetile; i++) {
        top += fitnesses.at(i);
        buttom += fitnesses.at(fitnesses.size() - 1 - i);
    }

    if (fitnesses.size()) {
        mean /= fitnesses.size();
    } else {
        mean = 0.0;
    }

    if (percetile) {
        top /= percetile;
        buttom /= percetile;
    } else {
        top = 0.0;
        buttom = 0.0;
    }

    stream << top << "," << buttom << "," << mean << "," << unimproved_counter << std::endl;
}
