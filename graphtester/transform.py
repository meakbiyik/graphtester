"""Pretransform methods for the DataLoader.

Non-private methods in this module needs to be serializeable.
"""
from typing import List, Union

import igraph as ig

import graphtester as gt
from graphtester.label import EDGE_LABELING_METHODS, VERTEX_LABELING_METHODS


def pretransform(  # noqa: C901
    features: List[str],
    feature_names: Union[str, List[str]] = None,
    encode: bool = False,
    encode_together: bool = False,
):
    """Transform data to contain desired feature(s).

    It either adds all features as a multidimensional tensor to the data object,
    or it encodes all the features as a single integer.

    Data is placed under the key with the name of the feature, with an underscore
    prepended. For example, if the feature is "Edge betweenness", the data will be
    placed under the key "_edge_betweenness". To override this behavior, the feature
    names can be specified using the feature_names argument, which should be a single
    string, or a list of strings of the same length as features. If there are
    overlapping feature names, or if the argument is a single string, the data will
    be concatenated. For example, if feature_names="x", then the data will be appended
    to the x tensor, if it exists, or a new x tensor will be created, so that it can
    be accessed via data.x. Note that this only works if the features are all
    edge or vertex features, otherwise a RuntimeError will be raised.

    If encode=True and encode_together=True, the data will be placed under the key
    provided (as a string) in the argument feature_names, by default "_encoding".

    For some varying-size multidimensional features such as subconstituent signatures,
    using multidimensional tensors is not possible (at least in a general and efficient
    way), so they can only be encoded as integers for the time being. If a feature
    in the list cannot be embedded as a tensor while encode=False, an error will be
    raised.

    encode=True should probably be used together with an embedding layer.

    Parameters
    ----------
    features : List[str]
        List of features to add to the data object.
    feature_names : str | List[str], optional
        List of names to use for the features, or a single feature name to
        add the data altogether, by default None
    encode : bool, optional
        Whether to encode the features as a single integer, by default False
    encode_together : bool, optional
        Whether to encode all features into a single integer, by default False.
        Does not have any effect if encode=False.

    Returns
    -------
    callable
        The pretransform method.
    """
    import torch  # noqa: F401

    hashmap = {}

    if feature_names is None:
        if encode and encode_together:
            feature_names = ["_encoding"] * len(features)
        else:
            feature_names = [f"_{f}".replace(" ", "_").lower() for f in features]
    elif isinstance(feature_names, str):
        feature_names = [feature_names] * len(features)

    if len(feature_names) != len(features):
        raise ValueError(
            "The number of feature names must be the same as the number of features."
        )

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
                    _push_to_data(data, feature_name, vals)

            else:
                try:
                    vals = [float(v) for v in vals]
                except ValueError:
                    raise ValueError(
                        f"Feature '{feature}' cannot be embedded as a tensor. "
                        "Try using encode=True."
                    )
                _push_to_data(data, feature_name, vals)

        if encode and encode_together:
            for v in zip(*value_lists):
                if v not in hashmap:
                    hashmap[v] = len(hashmap)
            vals = [hashmap[v] for v in zip(*value_lists)]
            _push_to_data(data, feature_names[0], vals)

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


def _push_to_data(data, key, value):
    import torch  # noqa: F401

    if key in data:
        value = torch.tensor(value).view(-1, 1)
        data[key] = torch.cat((data[key].view(value.shape[0], -1), value), 1)
    else:
        data[key] = torch.tensor(value).view(-1, 1)
