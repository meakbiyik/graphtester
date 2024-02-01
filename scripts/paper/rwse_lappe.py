import graphtester as gt
from matplotlib import pyplot as plt
import multiprocessing as mp

# Load ZINC
zinc = gt.load("ZINC", graph_count=10000)

baseline = gt.evaluate(zinc, metrics=["lower_bound_mse_graph_node"])
print("Baseline:", baseline.as_dataframe().iloc[3])


lappe_features = [
    "Laplacian positional encoding (dim=1)",
    "Laplacian positional encoding (dim=2)",
    "Laplacian positional encoding (dim=4)",
    "Laplacian positional encoding (dim=8)",
    "Laplacian positional encoding (dim=16)",
    "Laplacian positional encoding (dim=32)",
    "Laplacian positional encoding (dim=64)",
    "Laplacian positional encoding (dim=128)",
]

rwse_features = [
    "Random walk structural encoding (steps=1)",
    "Random walk structural encoding (steps=2)",
    "Random walk structural encoding (steps=4)",
    "Random walk structural encoding (steps=8)",
    "Random walk structural encoding (steps=16)",
    "Random walk structural encoding (steps=32)",
    "Random walk structural encoding (steps=64)",
    "Random walk structural encoding (steps=128)",
]

rrwp_features = [
    "Relative random walk probabilities (steps=1)",
    "Relative random walk probabilities (steps=2)",
    "Relative random walk probabilities (steps=4)",
    "Relative random walk probabilities (steps=8)",
    "Relative random walk probabilities (steps=16)",
    "Relative random walk probabilities (steps=32)",
    "Relative random walk probabilities (steps=64)",
    "Relative random walk probabilities (steps=128)",
]

spdpe_feature = "Shortest path distance positional encoding"

# plot the results with x axis as the dimension of the features
# and y axis as the accuracy, with the different lines being the
# different feature types, and baseline as a horizontal line

def evaluate(feature):
    print("evaluating", feature)
    return gt.evaluate(
        zinc,
        metrics=["lower_bound_mse_graph_node"],
        additional_features=[feature]).as_dataframe().iloc[3]

with mp.Pool(4) as pool:
    lappe_results = pool.map(evaluate, lappe_features)

with mp.Pool(4) as pool:
    rwse_results = pool.map(evaluate, rwse_features)

with mp.Pool(4) as pool:
    rrwp_results = pool.map(evaluate, rrwp_features)

spdpe_result = evaluate(spdpe_feature)

print("LapPE:", lappe_results)
print("RWSE:", rwse_results)
print("RRWP:", rrwp_results)
print("SPDPE:", spdpe_result)

# plot the results
mx = 12
# style: whitegrid
plt.style.use("seaborn-whitegrid")
# use serif font
plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(6, 2))
plt.title("ZINC")
plt.xlabel("Feature dimension")
plt.ylabel("MSE lower bound")
# use colors c2daf0 f9ceb1
plt.plot([1, 2, 4, 8, 16, 32, 64, 128][:mx], [result.item() for result in lappe_results][:mx], label="LapPE", marker="o", color="#77aeef")
plt.plot([1, 2, 4, 8, 16, 32, 64, 128][:mx], [result.item() for result in rwse_results][:mx], label="RWSE", marker="o", color="#fbb997")
# plt.plot([1, 2, 4, 8, 16, 32, 64, 128][:mx], [result.item() for result in rrwp_results][:mx], label="RRWP", marker="o", color="#b9fb97")
# plt.axhline(y=spdpe_result.item(), color="#ef79ef", linestyle="-", label="SPDPE")
plt.axhline(y=baseline.as_dataframe().iloc[3].item(), color="k", linestyle="--", label="Baseline")
plt.yscale("log")
plt.xscale("log", base=2)
plt.ylim([0.0001, 10])
# grid
plt.grid(True, which="both")
# only major ticks
plt.minorticks_off()
plt.legend()
# save as pdf
plt.savefig("rwse_lappe_zinc.pdf", bbox_inches="tight")
