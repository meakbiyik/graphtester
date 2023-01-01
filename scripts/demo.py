"""Demo script for the graphtester package."""
import graphtester as gt

datasets_to_evaluate = [
    "AIDS",
    "BZR",
    "COX2",
    "DHFR",
    "MUTAG",
    "NCI1",
    "NCI109",
    "ENZYMES",
    "DD",
    "PROTEINS",
]

for dataset_name in datasets_to_evaluate:
    print(f"Dataset: {dataset_name}")

    # Load the dataset
    dataset = gt.load(dataset_name)
    print(dataset)

    # Evaluate the dataset
    results = gt.evaluate(dataset)
    print(results)

    if results.identifiability[1] == 1:
        print("Dataset is fully identifiable")
    else:
        # Recommend features to add to the dataset
        recommendation = gt.recommend(dataset)

        # Print the recommendation
        print(recommendation.as_dataframe())
