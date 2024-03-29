"""Demo script for the graphtester package."""
import graphtester as gt

dataset_names = [
    "Cora",
    "Citeseer",
]

for dataset_name in dataset_names:
    print(f"Dataset: {dataset_name}")

    # Load the dataset
    dataset = gt.load(dataset_name)
    print(dataset)

    metrics = ["upper_bound_accuracy_node", "upper_bound_f1_micro_node"]

    # Evaluate the dataset
    evaluation = gt.evaluate(dataset, metrics=metrics)
    print(evaluation)

    if (
        evaluation.results["upper_bound_accuracy_node"][1] == 1
        and evaluation.results["upper_bound_f1_micro_node"][1] == 1
    ):
        print("Dataset is fully identifiable in one step, with node features.")
    else:
        # Recommend features to add to the dataset
        recommendation = gt.recommend(dataset, metrics=metrics, max_feature_count=3)

        # Print the recommendation
        print(recommendation.as_dataframe())
