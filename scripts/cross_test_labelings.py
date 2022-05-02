"""Cross-test labelings with all graph classes."""
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd

from graphtester import (
    ALL_METHODS,
    FAST_GRAPH_CLASSES,
    GRAPH_CLASS_DESCRIPTIONS,
    GRAPH_CLASSES,
    METHOD_DESCRIPTIONS,
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
        ("nbhood_subgraph_comp_sign", "edge_betweenness"),
    ]
    max_graph_count = 20
    process_count = 1

else:
    classes_to_test = GRAPH_CLASSES
    methods_to_test = ALL_METHODS + [
        ("nbhood_subgraph_comp_sign", "edge_betweenness"),
    ]
    max_graph_count = None
    process_count = 4  # If 1, the multiprocessing will be disabled.


def run_all_tests():
    """Run all tests for the given classes/methods.

    Returns
    -------
        None
    """
    print(classes_to_test)
    print(methods_to_test)

    method_descriptions = {
        method: METHOD_DESCRIPTIONS[method]
        if isinstance(method, str)
        else " + ".join(METHOD_DESCRIPTIONS[m] for m in method)
        for method in methods_to_test
    }

    all_graphs = {cls: get_graphs(cls) for cls in classes_to_test}

    columns = pd.MultiIndex.from_tuples(
        [
            [GRAPH_CLASS_DESCRIPTIONS[cls], ncount]
            for cls, ncounts in classes_to_test.items()
            for ncount in ncounts
        ]
        + [("Total", "")],
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
    rows["Graph class size"] = graph_class_sizes + [sum(graph_class_sizes)]
    rows["Test count n(n−1)/2"] = wl_test_counts + [sum(wl_test_counts)]

    vanilla_results, vanilla_failures = evaluate_method(
        all_graphs, "vanilla", return_failed_tests=True, max_graph_count=max_graph_count
    )
    rows["Vanilla 1-WL"] = vanilla_results + [sum(vanilla_results)]

    start_time = time.perf_counter()

    if process_count == 1:
        for method in methods_to_test:
            results = evaluate_method(
                all_graphs,
                method,
                graph_pair_indices=vanilla_failures,
                max_graph_count=max_graph_count,
            )
            rows[method_descriptions[method]] = results + [sum(results)]
    else:
        pool = mp.Pool(process_count)
        results = pool.starmap(
            evaluate_method,
            [
                (all_graphs, method, max_graph_count, vanilla_failures)
                for method in methods_to_test
            ],
        )
        pool.close()
        pool.join()
        for method, result in zip(methods_to_test, results):
            rows[method_descriptions[method]] = result + [sum(result)]

    print(f"Time spent: {round(time.perf_counter() - start_time, 2)}s")

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
    s.set_td_classes(cell_color)

    RESULTS_DIR.mkdir(exist_ok=True)
    s.to_html(
        RESULTS_DIR / f"wl_tests_{len(classes_to_test)}.html",
        table_attributes='border="0" cellspacing="0" cellpadding="4px"',
    )

    results_df.to_string(RESULTS_DIR / f"wl_tests_{len(classes_to_test)}.csv")


if __name__ == "__main__":
    run_all_tests()
