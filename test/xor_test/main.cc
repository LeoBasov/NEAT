#include <iostream>
#include <fstream>

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
    config.n_phenotypes = 150;

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
        std::cout << "ITERATION COMPLETE: " << i + 1 << "/" << n_itarations << std::endl;
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

    std::ofstream stream("genes.csv");

    for(auto gene : neat.gene_pool_.genes_){
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
    VectorXd input1(3), input2(3), input3(3), input4(3);
    VectorXd output1(1), output2(1), output3(1), output4(1);
    std::vector<std::pair<VectorXd, VectorXd>> input_outputs;

    input1(0) = 1.0;
    input1(1) = 1.0;
    input1(2) = 1.0;

    input2(0) = 1.0;
    input2(1) = 0.0;
    input2(2) = 1.0;

    input3(0) = 1.0;
    input3(1) = 1.0;
    input3(2) = 0.0;

    input4(0) = 1.0;
    input4(1) = 0.0;
    input4(2) = 0.0;

    output1(0) = 0.0;
    output2(0) = 1.0;
    output3(0) = 1.0;
    output4(0) = 0.0;

    input_outputs.push_back({input1, output1});
    input_outputs.push_back({input2, output2});
    input_outputs.push_back({input3, output3});
    input_outputs.push_back({input4, output4});

    return input_outputs;
}
