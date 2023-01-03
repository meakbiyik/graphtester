"""Pretransform methods for the DataLoader.

Non-private methods in this module needs to be serializeable.
"""
import igraph as ig

import graphtester as gt
from graphtester.label import EDGE_LABELING_METHODS, VERTEX_LABELING_METHODS


def pretransform(  # noqa: C901
    features: list[str],
    feature_names: list[str] = None,
    encode: bool = False,
    encode_together: bool = False,
    encoded_feature_name: str = "_encoding",
):
    """Transform data to contain desired feature(s).

    It either adds all features as a multidimensional tensor to the data object,
    or it encodes all the features as a single integer. For some varying-size
    multidimensional features such as subconstituent signatures, using
    multidimensional tensors is not possible (at least in a general and efficient
    way), so they can only be encoded as integers for the time being. If a feature
    in the list cannot be embedded as a tensor while encode=False, an error will be
    raised.

    Data is placed under the key with the name of the feature, with an underscore
    prepended. For example, if the feature is "degree", the data will be placed under
    the key "_degree". To override this behavior, the feature names can be specified
    using the feature_names argument, which should be a list of strings of the same
    length as features. If encode=True and encode_together=True, the data will be
    placed under the key provided in the argument encoded_feature_name, by default
    "_encoding".

    encode=True should probably be used together with an embedding layer.

    Parameters
    ----------
    features : list[str]
        List of features to add to the data object.
    feature_names : list[str], optional
        List of names to use for the features, by default None
    encode : bool, optional
        Whether to encode the features as a single integer, by default False
    encode_together : bool, optional
        Whether to encode all features into a single integer, by default False.
        Does not have any effect if encode=False.
    encoded_feature_name : str, optional
        The name to use for the encoded feature, by default "_encoding"

    Returns
    -------
    callable
        The pretransform method.
    """
    import torch  # noqa: F401

    hashmap = {}

    if feature_names is None:
        feature_names = [f"_{f}" for f in features]

    def _feature_estimator(data):
        graph = _to_igraph(data)

        value_lists = []
        for feature, feature_name in zip(features, feature_names):
            vals = _get_feature_values(graph, feature)

            if encode:
                if encode_together:
                    value_lists.append(vals)
                else:
                    if feature not in hashmap:
                        hashmap[feature] = {v: i for i, v in enumerate(set(vals))}
                    else:
                        for v in vals:
                            if v not in hashmap[feature]:
                                hashmap[feature][v] = len(hashmap[feature])
                    vals = [hashmap[feature][v] for v in vals]
                    data[feature_name] = torch.tensor(vals).view(-1, 1)

            else:
                try:
                    vals = [float(v) for v in vals]
                except ValueError:
                    raise ValueError(
                        f"Feature '{feature}' cannot be embedded as a tensor. "
                        "Try using encode=True."
                    )
                data[feature_name] = torch.tensor(vals).view(-1, 1)

        if encode and encode_together:
            for v in zip(*value_lists):
                if v not in hashmap:
                    hashmap[v] = len(hashmap)
            vals = [hashmap[v] for v in zip(*value_lists)]
            data[encoded_feature_name] = torch.tensor(vals).view(-1, 1)

        return data

    return _feature_estimator


def _to_igraph(data, directed=False):
    ig_graph = ig.Graph(
        n=data.num_nodes, edges=data.edge_index.t().tolist(), directed=directed
    )
    return ig_graph


def _get_feature_values(graph, feature):
    if feature in VERTEX_LABELING_METHODS:
        vals = gt.label(graph, [feature], copy=True).vs["label"]
    elif feature in EDGE_LABELING_METHODS:
        vals = gt.label(graph, [feature], copy=True).es["label"]
    else:
        raise ValueError(f"Feature '{feature}' not recognized.")
    return vals
