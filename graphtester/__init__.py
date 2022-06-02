"""Generate random non-isomorphic graphs."""
from graphtester.label import (
    ALL_METHODS,
    EDGE_LABELING_METHODS,
    EDGE_REWIRING_METHODS,
    VERTEX_LABELING_METHODS,
    label_graph,
)
from graphtester.produce import (
    FAST_GRAPH_CLASSES,
    GRAPH_CLASS_DESCRIPTIONS,
    GRAPH_CLASSES,
    get_graphs,
)
from graphtester.test import (
    k_weisfeiler_lehman_test,
    weisfeiler_lehman_hash,
    weisfeiler_lehman_test,
)

from graphtester.evaluate import evaluate_method  # isort:skip
