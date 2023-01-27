"""Demo script for the graphtester package."""
import graphtester as gt

dataset_names = [
    "ogbg-molesol",
    "ogbg-molfreesolv",
    "ogbg-mollipo",
]

for dataset_name in dataset_names:
    print(f"Dataset: {dataset_name}")

    # Load the dataset
    dataset = gt.load(dataset_name)
    print(dataset)

    metrics = ["lower_bound_mse"]

    # Evaluate the dataset
    evaluation = gt.evaluate(dataset, metrics=metrics)
    print(evaluation)

    if evaluation.results["lower_bound_mse"][1] == 1:
        print("Dataset is fully identifiable in one step, with node features.")
    else:
        # Recommend features to add to the dataset
        recommendation = gt.recommend(dataset, metrics=metrics, max_feature_count=3)

        # Print the recommendation
        print(recommendation.as_dataframe())
