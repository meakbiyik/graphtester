"""Plots for the graphtester package."""
import graphtester as gt

from matplotlib import pyplot as plt

dataset_names_gr = [
    "ogbg-molesol",
    "ogbg-molfreesolv",
    "ogbg-mollipo",
]

results_list_gr = []
for dataset_name in dataset_names_gr:
    dataset = gt.load(dataset_name)
    metrics = ["lower_bound_mse"]
    evaluation = gt.evaluate(dataset, metrics=metrics)
    print(evaluation)
    results = list(evaluation.results["lower_bound_mse"].values())
    results_list_gr.append(results)

fig = plt.figure(figsize=(8, 4))
for dataset_name, results in zip(dataset_names_gr, results_list_gr):
    plt.plot(
        results,
        label=dataset_name,
    )
plt.xlabel("Layer count")
plt.ylabel("Lower bound MSE")
plt.legend()
plt.xticks(range(len(results)))
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.title("Lower bound MSE vs layer count for graph regression datasets")
plt.savefig("lower_bound_mse_vs_layer_count_for_graph_regression_datasets.png", bbox_inches='tight')
plt.show()

# graph classification
dataset_names_gc = [
    "Mutagenicity",
    "MUTAG",
    "NCI1",
    "NCI109",
    "PTC_FM",
]

results_list_gc = []
for dataset_name in dataset_names_gc:
    dataset = gt.load(dataset_name)
    metrics = ["upper_bound_accuracy"]
    evaluation = gt.evaluate(dataset, metrics=metrics)
    print(evaluation)
    results = list(evaluation.results["upper_bound_accuracy"].values())
    results_list_gc.append(results)
    
fig = plt.figure(figsize=(8, 4))
for dataset_name, results in zip(dataset_names_gc, results_list_gc):
    plt.plot(
        results,
        label=dataset_name,
    )
plt.xlabel("Layer count")
plt.ylabel("Upper bound accuracy")
plt.legend()
plt.xticks(range(len(results)))
plt.grid()
plt.tight_layout()
plt.title("Upper bound accuracy vs layer count for graph classification datasets")
plt.savefig("upper_bound_accuracy_vs_layer_count_for_graph_classification_datasets.png", bbox_inches='tight')
plt.show()

# mse change per recommended feature at zero iteration
dataset_names = [
    "ogbg-molfreesolv",
]
recommendation_results_list_gc = {}
for idx, dataset_name in enumerate(dataset_names):
    dataset = gt.load(dataset_name)
    metrics = ["lower_bound_mse"]
    recommendation = gt.recommend(dataset, metrics=metrics, max_feature_count=1, fast=False)
    print(recommendation)
    for (state, features), results in recommendation.results.items():
        if state == "With node and edge features":
            recommendation_results_list_gc[features[0] if features else "Original"] = (
                results.results["lower_bound_mse"][0]
            )

# sort by mse change
recommendation_results_list_gc = {
    k: v
    for k, v in sorted(
        recommendation_results_list_gc.items(),
        key=lambda item: item[1],
    )
}

fig = plt.figure(figsize=(12, 6))
plt.bar(
    recommendation_results_list_gc.keys(),
    recommendation_results_list_gc.values(),
    # color origial differetly, the rest the same
    color=["C0" if key == "Original" else "C1" for key in recommendation_results_list_gc.keys()],
)
plt.xlabel("Recommended feature")
plt.ylabel("Lower bound MSE")
plt.xticks(rotation=90)
plt.grid()
plt.tight_layout()
plt.title("Lower bound MSE per recommended feature at zero iteration for graph regression datasets")
plt.savefig("lower_bound_mse_per_recommended_feature_at_zero_iteration_for_graph_regression_datasets.png", bbox_inches='tight')
plt.show()
