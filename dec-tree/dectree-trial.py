"""
Steps of CART Algorithms:
1. Select the Root node(S) based on the Gini Index and Highest Information Gain.
2. On each iteration of algorithms it calculates the Information gain, considering that every node never uses before.
3. Select the root node base on Highest I.G.
4. then splits set S to produce the subsets of data.
5. An algorithm continuous recurring on each subset and make sure that attributes are fresh and Create the decision Tree.
"""
"""
Info Gain ( S,A ) = Entropy(S) - sum(Sv / S * Entropy(Sv) for v in FeatureSet)

Entropy = -1 * sum(pi * log(pi) for i in c) # where pi is the prob of i.
1. Probability(Both Class) = 0.5 & Entropy = 1.
2. Probability(Either or Both Class) = 0 & Entropy = 0 it is called Leaf Node & stop Split.

Gini Index = 1 - sum( pi**2 for pi in c)
1. Lesser the value of the Gini index, the Higher the homogeneity.
2. 0 <= Gini Index <= 1
"""

import pandas as pandas
import math


def entire_entropy(column):
    entropy = 0
    my_set = {}
    for entry in column:
        if entry in my_set:
            my_set[entry] += 1
        else:
            my_set[entry] = 1
    length = len(column)
    for entry in my_set:
        weighted_avg = my_set[entry] / length
        entropy -= weighted_avg * math.log(weighted_avg, 2)

    return entropy


def info_gain(column, decision_column, entropy):
    # ########
    # Demo:
    # ########
    # col   decCol
    # a     0
    # a     1
    # b     0
    # c     1
    # a     0
    # b     0
    # c     1
    # ########
    # set will be created such that my_set = { a: {0:2,1:1}, b: {0:2,1:0}, c: {0:0,1:2}}
    # ########
    info_gain = 0
    my_set = {}
    for entry1, entry2 in zip(column, decision_column):
        if entry1 in my_set:
            if entry2 in my_set[entry1]:
                my_set[entry1][entry2] += 1
            else:
                my_set[entry1][entry2] = 1
        else:
            my_set[entry1] = {entry2: 1}
    # ########
    # Calculate entropy of each feature,
    # then multiply this "little_entropy" with their weights.
    # ########
    # Congrats, sum is Information Gain.
    # ########
    # Demo continued:
    # ########
    # entropy(a) = - 2/3 * log(2/3) - 1/3 * log(1/3)
    # entropy(b) = - 2/2 * log(2/2) - 0/2 * log(0/2)
    # entropy(c) = - 0/2 * log(0/2) - 2/2 * log(2/2)
    # info_gain = 3/7 * entropy(a) + 2/7 * entropy(b) + 2/7 * entropy(c)
    # ########
    for key in my_set:
        # print(my_set[key])
        # print(sum(my_set[key][x] for x in my_set[key]))
        occurrence_of_key = sum(my_set[key][x] for x in my_set[key])
        little_entropy = 0
        for key_inner in my_set[key]:
            weight = my_set[key][key_inner] / occurrence_of_key
            # print(weight)
            little_entropy -= weight * math.log(weight, 2)
        info_gain += occurrence_of_key / len(column) * little_entropy
    # print(my_set)
    return entropy - info_gain


def select_root(dataset):
    # print(dataset.iloc[:, -1])
    entropy = entire_entropy(dataset.iloc[:, -1])
    highest_info_gain = -1
    highest_info_column = ""
    for column in dataset.iloc[:, :-1]:
        # print()
        # print(dataset[column])
        # print(column, info_gain(dataset[column], dataset.iloc[:, -1],entropy))
        current_gain = info_gain(dataset[column], dataset.iloc[:, -1], entropy)
        # print(current_gain)
        if current_gain > highest_info_gain:
            highest_info_gain = current_gain
            highest_info_column = column
    # print(highest_info_column)
    return highest_info_column


def homogeneous(column):
    return len(set(column)) == 1


# def tree(dataset):
#     root = select_root(dataset)
#     my_set = set(dataset[root])
#     dataset_subsets = []
#     for feature in my_set:
#         # print(dataset[dataset[root] == feature])
#         dataset_subsets.append(dataset[dataset[root] == feature])
#
#     for node in dataset_subsets:
#         # print(node)
#         # print(node.iloc[:, -1])
#         if not homogeneous(node.iloc[:, -1]):
#             tree(node)


dataset = pandas.read_csv("weather.csv")
# tree(dataset)
