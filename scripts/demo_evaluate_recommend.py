"""Demo script for the graphtester package."""
import multiprocessing as mp

import graphtester as gt

MULTIPROCESSING = True

datasets_to_evaluate = [
    "BZR_MD",
    "COX2_MD",
    "DHFR_MD",
    "ER_MD",
    "Mutagenicity",
    "MUTAG",
    "NCI1",
    "NCI109",
    "PTC_FM",
    "PTC_FR",
    "PTC_MM",
    "PTC_MR",
    "ENZYMES",
    "DD",
    "PROTEINS",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "ogbg-molbace",
    "ogbg-molclintox",
    "ogbg-molbbbp",
    "ogbg-molsider",
    "ogbg-moltox21",
    "ogbg-moltoxcast",
    "ogbg-molhiv",
    "ogbg-molmuv",
    "FRANKENSTEIN",
    "COLLAB",
    "REDDIT-MULTI-5K",
]


def analyze_dataset(dataset_name: str):
    """Analyze a dataset."""
    print(f"Dataset: {dataset_name}")

    # Load the dataset
    dataset = gt.load(dataset_name)
    print(dataset)

    metrics = ["upper_bound_accuracy"]

    # Evaluate the dataset
    evaluation = gt.evaluate(dataset, metrics=metrics)
    print(evaluation)

    # Print the evaluation result to html
    evaluation.as_dataframe().to_html(f"evaluation_{dataset_name}.html")

    if evaluation.upper_bound_accuracy[1] == 1:
        print("Dataset is fully identifiable in one step, with node features.")
    else:
        # Recommend features to add to the dataset
        recommendation = gt.recommend(dataset, metrics=metrics, max_feature_count=3)

        # Print the recommendation
        print(recommendation.as_dataframe())

        # Print dataframe to html
        recommendation.as_dataframe().to_html(f"recommendation_{dataset_name}.html")


if __name__ == "__main__":

    if MULTIPROCESSING:
        with mp.Pool() as pool:
            pool.map(analyze_dataset, datasets_to_evaluate)

    else:
        for dataset_name in datasets_to_evaluate:
            analyze_dataset(dataset_name)
