#include <fstream>
#include <iostream>

#include "../../src/common/timer.h"
#include "../../src/neat/neat.h"
#include "../../src/neat/neat_algorithms.h"

using namespace neat;

std::vector<double> Execute(const NEAT &neat);
void PrintResultsNetwork(const NEAT& neat, const uint& network_id);
uint FindBestNetwork(const std::vector<double>& fitnesses);
void WriteNetworkToFile(const genome::Genotype& genotype, const GenePool& gene_pool,
                        const std::string& file_name = "best.csv");
void WriteFitnessToFile(std::ofstream& stream, std::vector<double> fitnesses, uint unimproved_counter);

int main(int, char**) {
    const uint n_iterations(2000);
    const double min_fitness(15.0);

    std::ofstream stream("fitness.csv");

    Timer exex, update, total;
    NEAT::Config config;
    genome::Genotype genotype;
    GenePool gene_pool;
    NEAT neat;
    const uint n_sensor_nodes(2), n_output_nodes(1), n_genotypes(150);
    std::vector<double> fitnesses;
    uint best_network_id;
    double mean(0.0);

    config.prob_new_node = 0.003;

    config.allow_recurring_connection = false;
    config.allow_self_connection = false;

    config.weight_range.first = -10.0;  // before -1000.0
    config.weight_range.second = 10.0;  // before  1000.0

    std::cout << "INITIALIZING" << std::endl;
    neat.Initialize(n_sensor_nodes, n_output_nodes, n_genotypes, config);

    total.Start();
    for (uint i = 0; i < n_iterations; i++) {
        exex.Start();
        fitnesses = Execute(neat);
        exex.Stop();

        best_network_id = FindBestNetwork(fitnesses);

        if (fitnesses.at(best_network_id) >= min_fitness) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "----- CRITERIUM ACHIEVED -----" << std::endl;
            std::cout << "BEST FITNESS: " << fitnesses.at(best_network_id) << std::endl;

            PrintResultsNetwork(neat, best_network_id);
            WriteNetworkToFile(neat.GetGenotypes().at(best_network_id), neat.GetGenePool());
            WriteFitnessToFile(stream, fitnesses, neat.GetUnimprovedCounter());

            break;
        }

        update.Start();
        neat.UpdateNetworks(fitnesses);
        update.Stop();

        mean = 0.0;

        for (auto fit : fitnesses) {
            mean += fit;
        }

        mean /= fitnesses.size();

        WriteFitnessToFile(stream, fitnesses, neat.GetUnimprovedCounter());

        if (!(i % 10)) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "ITTERATION: " << i << std::endl;
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << "EXECUTION TIME " << exex.GetCurrentDuration() << std::endl;
            std::cout << "UPDATE TIME " << update.GetCurrentDuration() << std::endl;
            std::cout << "N SPECIES   " << neat.GetSpecies().size() << std::endl;
            std::cout << "N GENOTYPES " << neat.GetGenotypes().size() << std::endl;
            std::cout << "N GENES     " << neat.GetGenePool().GetGenes().size() << std::endl;
            std::cout << "N NODES     " << neat.GetGenePool().GetNHiddenNodes() << std::endl;
            std::cout << "BEST FITNESS: " << fitnesses.at(best_network_id) << std::endl;
            std::cout << "MEAN FITNESS " << mean << std::endl;
        }
    }
    total.Stop();

    PrintResultsNetwork(neat, best_network_id);
    WriteNetworkToFile(neat.GetGenotypes().at(best_network_id), neat.GetGenePool(), "last.csv");

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

void WriteNetworkToFile(const genome::Genotype& genotype, const GenePool& gene_pool, const std::string& file_name) {
    std::ofstream stream(file_name);
    const std::map<uint, uint> permutation_map(neat_algorithms::GetPermutationMap(genotype));

    for (auto gene : genotype.genes) {
        if (gene.enabled) {
            GenePool::Gene gene_loc(gene_pool.GetGene(gene.id));

            stream << permutation_map.at(gene_loc.in_node) << "," << permutation_map.at(gene_loc.out_node) << ","
                   << gene.weight << std::endl;
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
