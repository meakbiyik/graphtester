"""Cross-test labelings with all graph classes."""
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List, Tuple

import igraph as ig
import pandas as pd

from graphtester import (
    ALL_METHODS,
    FAST_GRAPH_CLASSES,
    GRAPH_CLASS_DESCRIPTIONS,
    GRAPH_CLASSES,
    evaluate_method,
    get_graphs,
)

RESULTS_DIR = Path(__file__).parents[1] / "results"


def in_notebook():
    """Check if we are in a Jupyter notebook.

    Returns
    -------
    bool
        True if we are in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    classes_to_test = FAST_GRAPH_CLASSES
    methods_to_test = ALL_METHODS + [
        ("Neighborhood 1st subconstituent signatures", "Edge betweenness"),
        ("Neighborhood 2nd subconstituent signatures", "Edge betweenness"),
    ]
    max_node_count = 20
    max_graph_count = 20
    skip_3fwl = True
    process_count = 1
    silent = True

else:
    classes_to_test = GRAPH_CLASSES
    methods_to_test = ALL_METHODS + [
        ("Neighborhood 1st subconstituent signatures", "Edge betweenness"),
        ("Neighborhood 2nd subconstituent signatures", "Edge betweenness"),
    ]
    max_node_count = 30  # 40 is more extensive for a full test
    max_graph_count = None
    skip_3fwl = True  # Enable for final evaluation
    process_count = 16  # If 1, the multiprocessing will be disabled.
    silent = False

classes_to_test = {
    graph_class: [n for n in node_counts if n <= max_node_count]
    for graph_class, node_counts in classes_to_test.items()
}


def evaluate_and_time(
    all_graphs: List[ig.Graph],
    method: Tuple[str],
    max_graph_count: int,
    graph_pair_indices: Dict[str, List[Tuple[int, int]]],
):
    """Evaluate the labeling method and time it.

    Parameters
    ----------
    all_graphs : List[ig.Graph]
        The graphs to test.
    method : Tuple[str]
        The labeling method(s) to use.
    max_graph_count : int
        The maximum number of graphs to test.
    graph_pair_indices : Dict[str, List[Tuple[int, int]]]
        The indices of the graphs to test.

    Returns
    -------
    Tuple[int, float]
        The number of failed tests and the time spent in seconds.
    """
    start_time = time.perf_counter()
    results = evaluate_method(
        all_graphs,
        method,
        graph_pair_indices=graph_pair_indices,
        max_graph_count=max_graph_count,
        silent=silent,
    )
    time_spent = round(time.perf_counter() - start_time, 2)
    return results, time_spent


def run_all_tests():
    """Run all tests for the given classes/methods.

    Returns
    -------
        None
    """
    print(classes_to_test)
    print(methods_to_test)

    method_descriptions = {
        method: method if isinstance(method, str) else " + ".join(method)
        for method in methods_to_test
    }

    all_graphs = {cls: get_graphs(cls, max_node_count) for cls in classes_to_test}

    columns = pd.MultiIndex.from_tuples(
        [
            [GRAPH_CLASS_DESCRIPTIONS[cls], ncount]
            for cls, ncounts in classes_to_test.items()
            for ncount in ncounts
        ]
        + [("Total", ""), ("Time spent", "")],
        names=["graph_class: ", "node_count: "],
    )

    rows = {}
    # Add descriptive rows
    graph_class_sizes = [
        len(all_graphs[cls][ncount])
        for cls, ncounts in classes_to_test.items()
        for ncount in ncounts
    ]
    wl_test_counts = [int(cnt * (cnt - 1) / 2) for cnt in graph_class_sizes]
    rows["Graph class size"] = graph_class_sizes + [sum(graph_class_sizes), "-"]
    rows["Test count n(n−1)/2"] = wl_test_counts + [sum(wl_test_counts), "-"]

    vanilla_results, vanilla_failures = evaluate_method(
        all_graphs,
        "vanilla",
        return_failed_tests=True,
        max_graph_count=max_graph_count,
        silent=silent,
    )
    rows["Vanilla 1-WL"] = vanilla_results + [sum(vanilla_results), "-"]

    fwl_2_results, fwl_2_failures = evaluate_method(
        all_graphs,
        "vanilla",
        test_degree=2,
        graph_pair_indices=vanilla_failures,
        return_failed_tests=True,
        max_graph_count=max_graph_count,
        silent=silent,
    )
    rows["2-FWL"] = fwl_2_results + [sum(fwl_2_results), "-"]

    if process_count == 1:

        if not skip_3fwl:
            fwl_3_results = evaluate_method(
                all_graphs,
                "vanilla",
                test_degree=3,
                max_graph_count=max_graph_count,
                graph_pair_indices=fwl_2_failures,
                silent=silent,
            )
            rows["3-FWL"] = fwl_3_results + [sum(fwl_3_results), "-"]

        for method in methods_to_test:
            results, time_spent = evaluate_and_time(
                all_graphs,
                method,
                graph_pair_indices=vanilla_failures,
                max_graph_count=max_graph_count,
            )
            rows[method_descriptions[method]] = results + [
                sum(results),
                f"{time_spent}s",
            ]

    else:

        if not skip_3fwl:
            with mp.Pool(process_count) as pool:
                fwl_3_results_async = pool.apply_async(
                    evaluate_method,
                    (
                        all_graphs,
                        "vanilla",
                        3,
                        max_graph_count,
                        fwl_2_failures,
                        False,
                        silent,
                    ),
                )
                results_and_times = pool.starmap(
                    evaluate_and_time,
                    [
                        (all_graphs, method, max_graph_count, vanilla_failures)
                        for method in methods_to_test
                    ],
                )
                fwl_3_results = fwl_3_results_async.get()

            rows["3-FWL"] = fwl_3_results + [sum(fwl_3_results), "-"]

        else:
            with mp.Pool(process_count) as pool:
                results_and_times = pool.starmap(
                    evaluate_and_time,
                    [
                        (all_graphs, method, max_graph_count, vanilla_failures)
                        for method in methods_to_test
                    ],
                )

        for method, (result, time_spent) in zip(methods_to_test, results_and_times):
            rows[method_descriptions[method]] = result + [sum(result), f"{time_spent}s"]

    results_df = pd.DataFrame.from_dict(rows, columns=columns, orient="index")

    index_names = {
        "selector": ".index_name",
        "props": (
            "font-style: italic; color: darkgrey; "
            "font-weight:normal; text-align: right;"
        ),
    }
    header_colors = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #000066; color: white;",
    }
    centered_headers = {"selector": "th.col_heading", "props": "text-align: center;"}
    bigger_headers = {
        "selector": "th.col_heading.level0",
        "props": "font-size: 1.25em;",
    }
    wider_indices = {"selector": "th.row_heading", "props": "min-width: 275px;"}
    centered_bold_data = {
        "selector": "td",
        "props": "text-align: center; font-weight: bold; color: black;",
    }

    s = results_df.style.set_table_styles(
        [
            index_names,
            header_colors,
            centered_headers,
            bigger_headers,
            wider_indices,
            centered_bold_data,
        ]
    )

    class_line_styles = [
        {"selector": "th", "props": "border-left: 1px solid white"},
        {"selector": "td", "props": "border-left: 1px solid #000066"},
    ]
    line_between_classes = {
        class_name: class_line_styles
        for class_name in columns.to_frame(index=False)
        .groupby("graph_class: ", as_index=False)
        .min()
        .iloc[1:]
        .itertuples(index=False, name=None)
    }
    s.set_table_styles(line_between_classes, overwrite=False, axis=0)

    class_row_lines = [
        {"selector": "th", "props": "border-bottom: 1px solid white"},
        {"selector": "td", "props": "border-bottom: 1px solid #000066"},
    ]
    line_after_metadata = {
        "Test count n(n−1)/2": class_row_lines,
    }
    s.set_table_styles(line_after_metadata, overwrite=False, axis=1)

    s.set_table_styles(
        [  # create internal CSS classes
            {"selector": ".none", "props": "background-color: #cccccc;"},
            {"selector": ".best", "props": "background-color: #e6ffe6;"},
            {
                "selector": ".better",
                "props": (
                    "background: linear-gradient(135deg, #ffffe6 25%, "
                    "#e6ffe6 25%, #e6ffe6 50%, #ffffe6 50%, #ffffe6 75%, "
                    "#e6ffe6 75%, #e6ffe6 100%);"
                ),
            },
            {"selector": ".neutral", "props": "background-color: #ffffe6;"},
            {"selector": ".bad", "props": "background-color: #ffe6e6;"},
        ],
        overwrite=False,
    )
    cell_color = results_df.apply(
        lambda col: pd.Series(
            ["none", "none"]
            + col.iloc[2:]
            .map(
                lambda x: "best"
                if x == 0
                else "bad"
                if x == col.iloc[2:].max()
                else "better"
                if x == col.iloc[2:].min()
                else "neutral"
            )
            .tolist(),
            index=col.index,
        ),
    )
    cell_color[("Time spent", "")] = ["none"] * len(cell_color)
    s.set_td_classes(cell_color)

    RESULTS_DIR.mkdir(exist_ok=True)
    s.to_html(
        RESULTS_DIR / f"wl_tests_{len(classes_to_test)}_{round(time.time())}.html",
        table_attributes='border="0" cellspacing="0" cellpadding="4px"',
    )

    results_df.to_string(
        RESULTS_DIR / f"wl_tests_{len(classes_to_test)}_{round(time.time())}.csv"
    )


if __name__ == "__main__":
    run_all_tests()