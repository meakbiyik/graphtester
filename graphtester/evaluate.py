"""Run 1-WL tests for given labeling methods and graph classes."""
import os
import time
from typing import Dict, List, Tuple, Union

import igraph as ig

from graphtester import k_weisfeiler_lehman_test as kwl_test
from graphtester import label_graph
from graphtester import weisfeiler_lehman_test as wl_test


def evaluate_method(
    graph_class_members: Dict[str, Dict[int, List[ig.Graph]]],
    labeling: Union[str, List[str]],
    test_degree: int = 1,
    max_graph_count: int = None,
    graph_pair_indices: List[Tuple[int, int]] = None,
    return_failed_tests: bool = False,
    silent: bool = False,
) -> Union[List[int], Tuple[List[int], Dict[str, List[Tuple[int, int]]]]]:
    """Run the 1-WL tests for the given labeling method over provided graphs.

    It is possible to provide graph_pair_indices to only run the tests for a
    subset of the graphs.

    This method checks whether a WL-test actually failed by checking the
    isomorphism of the graphs with bliss (see igraph.Graph.isomorphic)
    after a failed WL-test, as graph classes are not guaranteed to contain
    pairwise non-isomorphic graphs. For more information, see
    http://www.tcs.hut.fi/Software/bliss/benchmarks/index.shtml

    Note that this final isomorphism check will not be run if graph_pair_indices
    is provided, as then it is assumed that the WL-test is being run on some
    graphs that the vanilla 1-WL test has already failed on.

    Parameters
    ----------
    graph_class_members : Dict[str, Dict[int, List[ig.Graph]]]
        A dictionary of graph classes and their members, mapped per node count.
    labeling : Union[str, List[str]],
        The labeling method(s) to use.
    test_degree : int, optional
        The degree of the WL test. If 1, use 1-WL, if >1, use k-FWL.
    max_graph_count : int, optional
        The maximum number of graphs to test. If None (default), test all graphs.
    graph_pair_indices : List[Tuple[int,int]], optional
        The indices of the graphs to test. If None (default), test all graphs.
    return_failed_tests : bool, optional
        If True, return a list of failed tests.
    silent : bool, optional
        If True, don't print anything to stdout.

    Returns
    -------
    result : List[int]
        The number of failed tests for the given labeling method.
    failures : Tuple[List[int], Dict[str, List[Tuple[int, int]]]]
        Failed tests, as a dictionary with keys as the graph class - vertex
        count in the form f"{cls}_{vcount}", mapping to a list of failed
        test indices. Only present if return_failed_tests is True.
    """
    max_graph_count = 1e12 if max_graph_count is None else max_graph_count
    labeling = (labeling,) if isinstance(labeling, str) else labeling

    print_kwargs = dict(
        src=f"Pr.{os.getpid()}-{labeling[0] if len(labeling) == 1 else labeling}",
        silent=silent,
    )

    failures = (
        {
            f"{cls}_{vcount}": []
            for cls, count_maps in graph_class_members.items()
            for vcount in count_maps.keys()
        }
        if return_failed_tests
        else None
    )
    result = []

    for cls, count_maps in graph_class_members.items():

        _printer(msg=f"Testing graph class {cls}...", level=1, **print_kwargs)

        for vcount in count_maps.keys():

            _printer(msg=f"Testing {cls}_{vcount}...", level=2, **print_kwargs)

            graphs = graph_class_members[cls][vcount]

            if len(graphs) > max_graph_count:
                graphs = graphs[:max_graph_count]

            if labeling != ("vanilla",):
                if graph_pair_indices is not None:
                    relevant_graphs = set(
                        sum(graph_pair_indices[f"{cls}_{vcount}"], ())
                    )
                    graphs = [
                        label_graph(g, labeling) if idx in relevant_graphs else g
                        for idx, g in enumerate(graphs)
                    ]
                else:
                    graphs = [label_graph(g, labeling) for g in graphs]

            if graph_pair_indices is not None:
                fail_count, class_failures = _test_indexed_graphs(
                    graphs, graph_pair_indices[f"{cls}_{vcount}"], labeling, test_degree
                )

            else:
                fail_count, class_failures = _test_all_graphs(
                    graphs, labeling, test_degree
                )

            if return_failed_tests:
                failures[f"{cls}_{vcount}"] = class_failures

            result.append(fail_count)

            _printer(
                msg=f"{cls}_{vcount}: {fail_count} failed",
                level=2,
                **print_kwargs,
            )

    if return_failed_tests:
        return result, failures

    return result


def _printer(
    src: str, msg: str, level: int = 0, timestamp: bool = True, silent: bool = False
) -> None:
    if not silent:
        print(
            (
                (f"{time.strftime('%H:%M:%S', time.gmtime())} " if timestamp else "")
                + f"[{src}]{'-' * level} {msg}"
            ),
            flush=True,
        )


def _test_all_graphs(graphs: List[ig.Graph], labeling: Tuple[str], test_degree: int):
    """Run the 1-WL tests for the given graphs and labeling method.

    Parameters
    ----------
    graphs : List[ig.Graph]
        The graphs to test.
    labeling : Tuple[str]
        The labeling method(s) to use.
    test_degree : int
        The degree of the WL test. If 1, use 1-WL, if >1, use k-FWL.

    Returns
    -------
    Tuple[int, List[Tuple[int, int]]]
        The number of failed tests and the indices of failure.
    """
    fail_count = 0
    failures = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            test = _run_test(graphs[i], graphs[j], labeling, test_degree)
            if test and not graphs[i].isomorphic(graphs[j]):
                fail_count += 1
                failures.append((i, j))
    return fail_count, failures


def _test_indexed_graphs(
    graphs: List[ig.Graph],
    graph_pair_indices: List[Tuple[int, int]],
    labeling: Tuple[str],
    test_degree: int,
):
    """Run the 1-WL tests for the given graphs/indices and labeling method.

    Parameters
    ----------
    graphs : List[ig.Graph]
        The graphs to test.
    graph_pair_indices : List[Tuple[int,int]]
        The indices of the graphs to test.
    labeling : Union[str, List[str]]
        The labeling method(s) to use.
    test_degree : int
        The degree of the WL test. If 1, use 1-WL, if >1, use k-FWL.

    Returns
    -------
    int
        The number of failed tests.
    """
    fail_count = 0
    failures = []
    for (i, j) in graph_pair_indices:
        test = _run_test(graphs[i], graphs[j], labeling, test_degree)
        if test:
            fail_count += 1
            failures.append((i, j))
    return fail_count, failures


def _run_test(
    graph1: ig.Graph, graph2: ig.Graph, labeling: Tuple[str], test_degree: int
) -> bool:
    """Run the 1-WL test for the given graphs and labeling method.

    Parameters
    ----------
    graph1 : ig.Graph
        The first graph to test.
    graph2 : ig.Graph
        The second graph to test.
    labeling : Tuple[str]
        The labeling method to use.
    test_degree : int
        The degree of the WL test. If 1, use 1-WL, if >1, use k-FWL.

    Returns
    -------
    bool
        Whether the test failed, i.e. 1-WL concluded that the
        graphs are isomorphic.
    """
    test_method = (
        wl_test
        if test_degree == 1
        else lambda *args, **kwargs: kwl_test(
            *args, **kwargs, k=test_degree, folklore=True
        )
    )

    if labeling == ("vanilla",):
        test = test_method(graph1, graph2)
    else:
        test = test_method(
            graph1,
            graph2,
            node_attr="label",
            edge_attr="label",
        )

    return test
