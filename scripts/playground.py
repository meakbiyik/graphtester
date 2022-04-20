"""Playground file for the tests."""
from graphtester import get_graphs
from graphtester import weisfeiler_lehman_test as wl_test

graphs_dict = get_graphs("planar_conn")
graphs = graphs_dict[7]

for i in range(len(graphs)):
    for j in range(i, len(graphs)):
        if i != j:
            test = wl_test(graphs[i], graphs[j])
            if test:
                print(i, j)
