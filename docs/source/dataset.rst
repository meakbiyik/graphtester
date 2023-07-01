Graphtester dataset
============================

Graphtester also comes with a synthetic graph dataset, referred to as `GT` in the package, that uniquely serves as a rigorous testing ground for the effectiveness of node and edge pre-coloring methods within the 1-Weisfeiler-Lehman (1-WL) framework. It can be loaded similarly to other datasets.

.. code-block:: python

    import graphtester as gt

    dataset = gt.load("GT")

Datasheet - Frequently Asked Questions
--------------------------------------

Motivation
^^^^^^^^^^

**For what purpose was the dataset created?**
    Dataset is created to assess the expressive power of node and edge features in the framework of 1-Weisfeiler-Lehman test. Our expected use case is to compare potential positional encodings for tasks on graph datasets that researchers use GNN and GT models on.

**Who created the dataset and on behalf of which entity?**
    This will be revealed upon acceptance.

**Who funded the creation of the dataset?**
    This will be revealed upon acceptance.

Composition
^^^^^^^^^^^

**What do the instances that comprise the dataset represent?**
    Dataset contains synthetic undirected graphs without any node and edge labels, that are known to belong certain graph classes of certain order.

**How many instances are there in total?**
    There are 55,340 graphs in the dataset in total. Graphs are not necessarily non-isomorphic, since some graph classes overlap with each other.

**Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?**
    Graphs are not randomly generate nor selected. For all graph classes other than distance-regular graphs, they exhaustively represent the graphs that belong to the given order of their class. For distance-regular graphs, we use the whole compilation in Bailey et al. (2019).

**What data does each instance consist of?**
    The data instances consist of iGraph objects (Csardi and Nepusz, 2006), without any node and edge features.

**Is there a label or target associated with each instance?**
    There are no label associated with the instances. The task is to be able to distinguish all pairs of graphs in a certain graph class and order, for all given graph classes. In total, task requires 225'930'287 million successful pairwise comparisons.

**Is any information missing from individual instances?**
    No.

**Are there recommended data splits (e.g., training, development/validation, testing)?**
    Dataset is to be consumed as-is, and does not require any training-test split since there is no concept of training or overfitting on exhaustive domains.

**Are there any errors, sources of noise, or redundancies in the dataset?**
    Not to the knowledge of the authors.

**Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?**
    Dataset is fully self-contained, and possible to regenerate from scratch if needed.

**Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor– patient confidentiality, data that includes the content of individuals’ non-public communications)?**
    No.

**Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?**
    No.
