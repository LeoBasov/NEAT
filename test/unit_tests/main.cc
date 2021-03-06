#include <gtest/gtest.h>

#include "common/utility.h"
#include "io/genome.h"
#include "io/mnist.h"
#include "neat/gene_pool.h"
#include "neat/neat.h"
#include "neat/neat_algorithms.h"
#include "neat_new/genome.h"
#include "neat_new/mutator.h"
#include "neat_new/neat.h"
#include "neat_new/network.h"
#include "neat_new/species_pool.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
