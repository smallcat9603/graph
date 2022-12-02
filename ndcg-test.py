# smallcat 221201

import numpy as np
import ndcg

rel_true = [0, 1, 0, 1, 2, 3, 3, 1, 0] # [item1 score, item2 score, item3 score, ...] 

rel_pred = [0, 3, 3, 1, 2] # [real score of predicted item with highest score, real score of predicted item with second highest score, real score of predicted item with third highest score, ...]

p = 5

form = "exp"

print(f'p nDCG@{p} ({form}): {ndcg.ndcg(rel_true, rel_pred, p, form)}')
