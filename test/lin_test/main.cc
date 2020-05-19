#include <fstream>
#include <iostream>

#include "../../src/neat/neat.h"

using namespace Eigen;

std::vector<std::pair<VectorXd, VectorXd>> SetUpInputOutput();

int main() {
    const std::vector<std::pair<VectorXd, VectorXd>> input_outputs(SetUpInputOutput());
    NEAT::NEAT neat;
    NEAT::NEAT::Config config;
    const uint n_itarations(1000);

    config.n_input = 3;
    config.n_output = 1;
    config.n_phenotypes = 1500;

    neat.Initialize(config);
    neat.BuildNetworks();
    neat.Execute(input_outputs);

    std::sort(neat.phenotypes_.rbegin(), neat.phenotypes_.rend());

    std::cout << "=========================================================" << std::endl;
    std::cout << "XOR TEST" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "NUMBER NETWORKS: " << config.n_phenotypes << std::endl;
    std::cout << "NUMBER ITERATIONS: " << n_itarations << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    for (uint i = 0; i < n_itarations; i++) {
        neat.Speciate();
        neat.Reproduce();
        neat.Mutate();
        neat.BuildNetworks();
        neat.Execute(input_outputs);

        std::sort(neat.phenotypes_.rbegin(), neat.phenotypes_.rend());

        std::cout << "BEST: " << neat.phenotypes_.front().fitness_ << std::endl;
        std::cout << "WORST: " << neat.phenotypes_.back().fitness_ << std::endl;
        std::cout << "NUMBER SPECIES: " << neat.species_.size() << std::endl;
        std::cout << "NUMBER GENOMES: " << neat.phenotypes_.size() << std::endl;
        std::cout << "NUMBER GENES: " << neat.gene_pool_.genes_.size() << std::endl;
        std::cout << "NUMBER NODES: " << neat.gene_pool_.nodes_.size() << std::endl;
        std::cout << "ITERATION COMPLETE: " << i + 1 << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

        if (neat.phenotypes_.front().fitness_ > 0.99) {
            break;
        }
    }

    std::cout << neat.Str(0);
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << neat.Str(neat.phenotypes_.size() - 1);
    std::cout << "FINISHED" << std::endl;
    std::cout << "=========================================================" << std::endl;

    VectorXd output(1);

    neat.ExecuteNetwork(0, input_outputs.front().first, output);

    std::cout << input_outputs.front().first(1) << ", " << input_outputs.front().first(2) << ", " << output(0)
              << std::endl;

    std::ofstream stream("genes.csv");

    for (auto gene : neat.gene_pool_.genes_) {
        stream << gene.in << "," << gene.out << std::endl;
    }

    stream.close();

    stream.open("best.csv");

    for (auto gene : neat.phenotypes_.front().genes_) {
        stream << neat.gene_pool_.genes_.at(gene.id).in << "," << neat.gene_pool_.genes_.at(gene.id).out << std::endl;
    }

    stream.close();

    return 0;
}

std::vector<std::pair<VectorXd, VectorXd>> SetUpInputOutput() {
    const uint N(100);
    std::vector<std::pair<VectorXd, VectorXd>> input_outputs;
    NEAT::Random random;
    VectorXd input(3);
    VectorXd output(1);

    for (uint i = 0; i < N; i++) {
        input(0) = 1.0;
        input(1) = random.RandomNumber();
        input(2) = random.RandomNumber();

        while (input(2) < input(1)) {
            input(1) = random.RandomNumber();
            input(2) = random.RandomNumber();
        }

        output(0) = 0.5 * (input(1) + input(2));

        input_outputs.push_back({input, output});
    }

    return input_outputs;
}
