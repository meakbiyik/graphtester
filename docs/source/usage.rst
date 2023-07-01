Usage
=====

Graphtester can load, label, analyze, and test datasets. The following example shows how to load a dataset, label it, and test it with a GNN.

Loading a dataset
-----------------

Graphtester package exposes a `load` function that can load a dataset from various formats, and convert it to a `Dataset` object, internal storage format of Graphtester. `load` function can take a dataset name, list of NetworkX or iGraph graphs, or a PyG/DGL dataset as input. The following example loads the `MUTAG` dataset.

.. code-block:: python

    from graphtester import load
    dataset = load('MUTAG')

Running 1-WL or k-WL on a graph pair
-------------------------------------

Graphtester can run 1-WL or k-WL on a pair of graphs. The following example runs 1-WL on the first two graphs in the dataset.

.. code-block:: python

    from graphtester import (
        weisfeiler_lehman_test as wl_test,
        k_weisfeiler_lehman_test as kwl_test
    )

    G1, G2 = dataset.graphs[:2]
    is_iso = wl_test(G1, G2)
    is_iso_kwl = kwl_test(G1, G2, k=4)

Computing upper score bounds for a dataset
------------------------------------------

Graphtester can compute upper score bounds for a dataset and task. For calculating the upper score bounds associated with a specified number of layers, we utilize the congruence of Graph Neural Networks (GNNs) and graph transformers with the 1-Weisfeiler-Lehman test (1-WLE), as established in the paper.

.. code-block:: python

    import graphtester as gt

    dataset = gt.load("ZINC")
    evaluation = gt.evaluate(dataset)
    print(evaluation.as_dataframe())
