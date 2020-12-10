#include <iostream>

#include "../../src/common/timer.h"
#include "../../src/neat/neat.h"
#include "../../src/neat/neat_algorithms.h"

using namespace neat;

std::vector<double> Execute(const NEAT &neat);
void PrintResultsNetwork(const NEAT& neat, const uint& network_id);
uint FindBestNetwork(const std::vector<double>& fitnesses);

int main(int, char**) {
    const uint n_iterations(1000);
    const double min_fitness(12.0);

    Timer exex, update, total;
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensor_nodes(2), n_output_nodes(1), n_genotypes(150);
    std::vector<double> fitnesses;
    uint best_network_id;

    config.prob_new_node = 0.003;

    std::cout << "INITIALIZING" << std::endl;
    neat.Initialize(n_sensor_nodes, n_output_nodes, n_genotypes, config);

    total.Start();
    for (uint i = 0; i < n_iterations; i++) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ITTERATION: " << i << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "EXECUTING" << std::endl;
        exex.Start();
        fitnesses = Execute(neat);
        exex.Stop();
        std::cout << "EXECUTION TIME " << exex.GetCurrentDuration() << std::endl;

        best_network_id = FindBestNetwork(fitnesses);

        if (fitnesses.at(best_network_id) >= min_fitness) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "----- CRITERIUM ACHIEVED -----" << std::endl;
            std::cout << "BEST FITNESS: " << fitnesses.at(best_network_id) << std::endl;

            PrintResultsNetwork(neat, best_network_id);

            break;
        }

        std::cout << "UPDATING" << std::endl;
        update.Start();
        neat.UpdateNetworks(fitnesses);
        update.Stop();
        std::cout << "UPDATE TIME " << update.GetCurrentDuration() << std::endl;

        std::cout << "N SPECIES   " << neat.GetSpecies().size() << std::endl;
        std::cout << "N GENOTYPES " << neat.GetGenotypes().size() << std::endl;
        std::cout << "N GENES     " << neat.GetGenePool().GetGenes().size() << std::endl;
        std::cout << "N NODES     " << neat.GetGenePool().GetNHiddenNodes() << std::endl;
        std::cout << "BEST FITNESS: " << fitnesses.at(best_network_id) << std::endl;

        double mean(0.0);

        for (auto fit : fitnesses) {
            mean += fit;
        }

        mean /= fitnesses.size();

        std::cout << "MEAN FITNESS " << mean << std::endl;
    }
    total.Stop();

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "TOTAL EXECUTION TIME " << exex.GetTotalDuration() << std::endl;
    std::cout << "TOTAL UPDATE TIME " << update.GetTotalDuration() << std::endl;
    std::cout << "TOTAL TIME " << total.GetTotalDuration() << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}

std::vector<double> Execute(const NEAT &neat) {
    std::vector<double> refs(4);
    std::vector<double> fitnesses(neat.GetGenotypes().size(), 0.0);
    std::vector<std::vector<double>> fitnesses_loc;
    std::vector<std::vector<double>> input_values(4);

    refs.at(0) = 0.0;
    refs.at(1) = 0.0;
    refs.at(2) = 1.0;
    refs.at(3) = 1.0;

    input_values.at(0) = {0.0, 0.0};
    input_values.at(1) = {1.0, 1.0};
    input_values.at(2) = {1.0, 0.0};
    input_values.at(3) = {0.0, 1.0};

    for (uint i = 0; i < 4; i++) {
        fitnesses_loc = neat.ExecuteNetworks(input_values.at(i));

        for (uint j = 0; j < neat.GetGenotypes().size(); j++) {
            fitnesses.at(j) += std::abs(refs.at(i) - fitnesses_loc.at(j).at(0));
        }
    }

    for (uint j = 0; j < neat.GetGenotypes().size(); j++) {
        fitnesses.at(j) = std::pow(4.0 - fitnesses.at(j), 2);
    }

    return fitnesses;
}

void PrintResultsNetwork(const NEAT& neat, const uint& network_id) {
    std::vector<std::vector<double>> fitnesses(4);

    fitnesses.at(0) = neat.ExecuteNetwork({0.0, 0.0}, network_id);
    fitnesses.at(1) = neat.ExecuteNetwork({1.0, 1.0}, network_id);
    fitnesses.at(2) = neat.ExecuteNetwork({1.0, 0.0}, network_id);
    fitnesses.at(3) = neat.ExecuteNetwork({0.0, 1.0}, network_id);

    std::cout << "INPUT 1: " << 0.0 << " INPUT 2: " << 0.0 << " OUTPUT: " << fitnesses.at(0).at(0) << std::endl;
    std::cout << "INPUT 1: " << 1.0 << " INPUT 2: " << 1.0 << " OUTPUT: " << fitnesses.at(1).at(0) << std::endl;
    std::cout << "INPUT 1: " << 1.0 << " INPUT 2: " << 0.0 << " OUTPUT: " << fitnesses.at(2).at(0) << std::endl;
    std::cout << "INPUT 1: " << 0.0 << " INPUT 2: " << 1.0 << " OUTPUT: " << fitnesses.at(3).at(0) << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "N NODES: " << neat.GetGenotypes().at(network_id).nodes.size();
    std::cout << " N GENES: " << neat.GetGenotypes().at(network_id).genes.size() << std::endl;
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
