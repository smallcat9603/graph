from typing import List, Set
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from scipy.optimize import linear_sum_assignment


# for disjoint communities

def normalized_mutual_information(a: List[Set[int]], b: List[Set[int]]):
    """_summary_

    Args:
        a (List[Set[int]]): list of communities
        b (List[Set[int]]): list of communities
    """
    nd2c_a, nd2c_b = [], []
    for i, s in enumerate(a):
        for nd in s:
            nd2c_a.append((nd, i))
    nd2c_a.sort()
    labels_true = [x[1] for x in nd2c_a]
    for i, s in enumerate(b):
        for nd in s:
            nd2c_b.append((nd, i))
    nd2c_b.sort()
    labels_pred = [x[1] for x in nd2c_b]
    return normalized_mutual_info_score(labels_true, labels_pred)


def f_score(a: List[Set[int]], b: List[Set[int]]):
    """_summary_

    Args:
        a (List[Set[int]]): list of communities
        b (List[Set[int]]): list of communities

    Return:
        F-score: f-score of 
    """
    nd2set_a, nd2set_b = dict(), dict()
    for s in a:
        for nd in s:
            nd2set_a[nd] = s
    for s in b:
        for nd in s:
            nd2set_b[nd] = s
    sum_f = 0
    for nd in nd2set_a.keys():
        f = compute_precision_recall_f1(nd2set_a[nd], nd2set_b[nd])[2]
        sum_f += f
    return sum_f / len(nd2set_a)


def edit_distance(a: List[Set[int]], b: List[Set[int]]):
    num_nodes = sum(len(s) for s in a)
    if len(a) > len(b):
        longer, shorter = a, b
    else:
        longer, shorter = b, a
    shorter += [set()] * (len(longer) - len(shorter))
    list_costs = []
    for sl in a:
        costs = []
        for ss in b:
            intersection = len(sl.intersection(ss))
            costs.append(intersection)
        list_costs.append(costs)
    cost_matrix = np.array(list_costs)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    return num_nodes - cost_matrix[row_ind, col_ind].sum()


# for overlapping communities

def f1(a: Set[int], b: Set[int]):
    return compute_precision_recall_f1(a, b)[2]


def relative_overlap(a: Set[int], b: Set[int]):
    return len(a.intersection(b)) / len(a.union(b))


def compute_precision_recall_f1(prediction, ground_truth: set):
    hit = 0  # number of true positive
    for p in prediction:
        if p in ground_truth:
            hit += 1
    if len(prediction) == 0:
        precision = 0
    else:
        precision = hit / len(prediction)
    if len(ground_truth) == 0:
        recall = 0
    else:
        recall = hit / len(ground_truth)
    if hit == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def unique_influence(incl: Set[int], excl: Set[int], B: Set[int]):
    nui: Set[int] = incl.intersection(base) - excl
    return len(nui) / len(base)


def relative_unique_influence(incl: Set[int], excl: Set[int], B: Set[int]):
    nui: Set[int] = incl.intersection(base) - excl
    return len(nui) / len(incl)
