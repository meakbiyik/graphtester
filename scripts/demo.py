"""Demo script for the graphtester package."""
import graphtester as gt

# Load the dataset
dataset = gt.load("MUTAG")
print(dataset)

# Evaluate the dataset
results = gt.evaluate(dataset, ignore_node_features=False, ignore_edge_features=True)
print(results)

# Recommend features to add to the dataset
recommendation = gt.recommend(dataset)

# Print the recommendation
print(recommendation)
