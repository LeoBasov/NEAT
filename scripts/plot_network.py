#!/usr/bin/python3

from graphviz import Digraph
import csv
import sys

dot = Digraph(comment='Network')

with open(sys.argv[1], newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in reader:
		dot.edge(row[0], row[1])

dot.render()
