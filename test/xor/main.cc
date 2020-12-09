#include <iostream>

#include "../../src/common/timer.h"
#include "../../src/neat/neat.h"
#include "../../src/neat/neat_algorithms.h"

using namespace neat;

std::vector<double> Execute(const NEAT &neat);

int main(int, char**) {
    Timer exex, update, total;
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensor_nodes(2), n_output_nodes(1), n_genotypes(150);
    std::vector<double> fitnesses;

    std::cout << "INITIALIZING" << std::endl;
    neat.Initialize(n_sensor_nodes, n_output_nodes, n_genotypes, config);

    total.Start();
    for (uint i = 0; i < 100; i++) {
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "ITTERATION: " << i << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "EXECUTING" << std::endl;
        exex.Start();
        fitnesses = Execute(neat);
        exex.Stop();
        std::cout << "EXECUTION TIME " << exex.GetCurrentDuration() << std::endl;

        if (fitnesses.front() > 1.75) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "FITNESS     " << fitnesses.front() << std::endl;
            Execute(neat);

            std::vector<std::vector<double>> fitnesses_loc = neat.ExecuteNetworks({0.0, 0.0});

            std::cout << "INPUT 1: " << 0.0 << " INPUT 2: " << 0.0 << " OUTPUT: " << fitnesses_loc.front().at(0)
                      << std::endl;

            fitnesses_loc = neat.ExecuteNetworks({1.0, 1.0});

            std::cout << "INPUT 1: " << 1.0 << " INPUT 2: " << 1.0 << " OUTPUT: " << fitnesses_loc.front().at(0)
                      << std::endl;

            fitnesses_loc = neat.ExecuteNetworks({1.0, 0.0});

            std::cout << "INPUT 1: " << 1.0 << " INPUT 2: " << 0.0 << " OUTPUT: " << fitnesses_loc.front().at(0)
                      << std::endl;

            fitnesses_loc = neat.ExecuteNetworks({0.0, 1.0});

            std::cout << "INPUT 1: " << 0.0 << " INPUT 2: " << 1.0 << " OUTPUT: " << fitnesses_loc.front().at(0)
                      << std::endl;

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
        std::cout << "FITNESS     " << fitnesses.front() << std::endl;
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
        fitnesses.at(j) = std::sqrt(4.0 - fitnesses.at(j));
    }

    return fitnesses;
}
