#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_fitness():
    print("sys_arg:", sys.argv[1])
    fitnesses = []

    with open(sys.argv[1], newline='') as csvfile:
        spamreader = csv.reader(csvfile)

        for row in spamreader:
            fitness_row = []

            for elem in row:
                try:
                    elem_f = float(elem)
                    fitness_row.append(elem_f)
                except Exception as err:
                    break

            fitnesses.append(fitness_row)

    means = []
    top_percentile = []
    perc = int(len(fitnesses[0])/10)

    for row in fitnesses:
        mean = 0.0

        row.sort()
        row.reverse()

        for i in range(len(row)):
            mean += row[i]

            if (i + 1)%perc == 0:
                break

        top_percentile.append(mean/perc)

    for row in fitnesses:
        mean = 0

        for elem in row:
            mean += elem

        means.append(mean/len(row))

    plt.plot(means)
    plt.plot(top_percentile)
    plt.legend(["mean", "top_percentile"])
    plt.show()

if __name__  == "__main__":
    plot_fitness()
