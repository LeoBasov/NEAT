#!/usr/bin/python3

from graphviz import Digraph
import csv

dot = Digraph(comment='Network')

with open('genes.csv', newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in reader:
		dot.edge(row[0], row[1])

dot.render()
