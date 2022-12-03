# smallcat 221201

import numpy as np
import ndcg

rel_true = [0, 1, 0, 1, 2, 3, 3, 1, 0] # [item0 score, item1 score, item2 score, ...] 

rank = [2, 6, 5, 1, 4] # [item predicted as first rank, item predicted as second rank, item predicted as third rank, ...]
rel_pred = [rel_true[i] for i in rank] # [real score of predicted item with highest score, real score of predicted item with second highest score, real score of predicted item with third highest score, ...]

p = len(rank)

form = "linear"

print(f'p nDCG@{p} ({form}): {ndcg.ndcg(rel_true, rel_pred, p, form)}')
