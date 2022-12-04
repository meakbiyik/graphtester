"""Cross-test labelings with all graph classes."""
import hashlib
import multiprocessing as mp
import pickle
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
    produce,
)

RESULTS_DIR = Path(__file__).parents[1] / "results"
CACHE_DIR = Path(__file__).parents[1] / "cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


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
        ("1st subconstituent signatures", "Edge betweenness"),
        ("2nd subconstituent signatures", "Edge betweenness"),
    ]
    max_node_count = 20
    max_graph_count = 20
    skip_3fwl = True
    process_count = 1
    silent = True

else:
    classes_to_test = GRAPH_CLASSES
    methods_to_test = ALL_METHODS + [
        ("1st subconstituent signatures", "Edge betweenness"),
        ("2nd subconstituent signatures", "Edge betweenness"),
    ]
    max_node_count = 40
    max_graph_count = None
    skip_3fwl = False  # Enable for final evaluation
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


def _method_hash(method):
    methodstr = ",".join(method) if isinstance(method, tuple) else method
    methodstr = methodstr + str(max_graph_count) + repr(classes_to_test)
    return hashlib.sha1(methodstr.encode("UTF-8")).hexdigest()


def _save_to_cache(datahash, data):
    """Pickle given data to cache."""
    with open(CACHE_DIR / f"{datahash}.pkl", "wb") as f:
        pickle.dump(data, f)


def _retrieve_from_cache(datahash):
    """Retrieve from cache with the given hash if available."""
    if (CACHE_DIR / f"{datahash}.pkl").exists():
        with open(CACHE_DIR / f"{datahash}.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    else:
        return None


def _cache_output(f, datahash, *args, **kwargs):
    data = _retrieve_from_cache(datahash)
    if data is None:
        data = f(*args, **kwargs)
        _save_to_cache(datahash, data)
    return data


def _evaluate_method_cached(datahash, *args, **kwargs):
    return _cache_output(evaluate_method, datahash, *args, **kwargs)


def _evaluate_and_time_cached(datahash, *args, **kwargs):
    return _cache_output(evaluate_and_time, datahash, *args, **kwargs)


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

    all_graphs = {
        cls: produce(cls, max(node_counts))
        for cls, node_counts in classes_to_test.items()
    }

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

    vanilla_results, vanilla_failures = _evaluate_method_cached(
        _method_hash("Vanilla 1-WL"),
        all_graphs,
        "vanilla",
        return_failed_tests=True,
        max_graph_count=max_graph_count,
        silent=silent,
    )
    rows["Vanilla 1-WL"] = vanilla_results + [sum(vanilla_results), "-"]

    if process_count == 1:

        fwl_2_results, fwl_2_failures = _evaluate_method_cached(
            _method_hash("2-FWL"),
            all_graphs,
            "vanilla",
            test_degree=2,
            graph_pair_indices=vanilla_failures,
            return_failed_tests=True,
            max_graph_count=max_graph_count,
            silent=silent,
        )
        rows["2-FWL"] = fwl_2_results + [sum(fwl_2_results), "-"]

        if not skip_3fwl:
            fwl_3_results = _evaluate_method_cached(
                _method_hash("3-FWL"),
                all_graphs,
                "vanilla",
                test_degree=3,
                max_graph_count=max_graph_count,
                graph_pair_indices=fwl_2_failures,
                silent=silent,
            )
            rows["3-FWL"] = fwl_3_results + [sum(fwl_3_results), "-"]

        for method in methods_to_test:
            results, time_spent = _evaluate_and_time_cached(
                _method_hash(method),
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

        with mp.Pool(process_count) as pool:
            fwl_2_results_async = pool.apply_async(
                _evaluate_method_cached,
                (
                    _method_hash("2-FWL"),
                    all_graphs,
                    "vanilla",
                    2,
                    max_graph_count,
                    vanilla_failures,
                    True,
                    silent,
                ),
            )

            results_and_times_async = pool.starmap_async(
                _evaluate_and_time_cached,
                [
                    (
                        _method_hash(method),
                        all_graphs,
                        method,
                        max_graph_count,
                        vanilla_failures,
                    )
                    for method in methods_to_test
                ],
            )

            fwl_2_results, fwl_2_failures = fwl_2_results_async.get()

            if not skip_3fwl:
                fwl_3_results_async = pool.apply_async(
                    _evaluate_method_cached,
                    (
                        _method_hash("3-FWL"),
                        all_graphs,
                        "vanilla",
                        3,
                        max_graph_count,
                        fwl_2_failures,
                        False,
                        silent,
                    ),
                )
                fwl_3_results = fwl_3_results_async.get()

            results_and_times = results_and_times_async.get()

        rows["2-FWL"] = fwl_2_results + [sum(fwl_2_results), "-"]
        if not skip_3fwl:
            rows["3-FWL"] = fwl_3_results + [sum(fwl_3_results), "-"]
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

    s.to_latex(
        RESULTS_DIR
        / f"wl_tests_{len(classes_to_test)}_{round(time.time())}_styled.tex",
        convert_css=True,
    )

    s.to_excel(
        RESULTS_DIR
        / f"wl_tests_{len(classes_to_test)}_{round(time.time())}_styled.xlsx"
    )

    results_df.to_string(
        RESULTS_DIR / f"wl_tests_{len(classes_to_test)}_{round(time.time())}.csv"
    )

    results_df.to_latex(
        RESULTS_DIR / f"wl_tests_{len(classes_to_test)}_{round(time.time())}.tex"
    )

    results_df.to_excel(
        RESULTS_DIR / f"wl_tests_{len(classes_to_test)}_{round(time.time())}.xlsx"
    )

    return results_df


if __name__ == "__main__":
    run_all_tests()
