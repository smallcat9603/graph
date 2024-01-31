import pandas as pd
import ndcg

# # test
# rel_true = [0, 1, 0, 1, 2, 3, 3, 1, 0] # [item0 score, item1 score, item2 score, ...] 
# rank = [2, 6, 5, 1, 4] # [item predicted as first rank, item predicted as second rank, item predicted as third rank, ...]
# rel_pred = [rel_true[i] for i in rank] # [real score of predicted item with highest score, real score of predicted item with second highest score, real score of predicted item with third highest score, ...]
# p = len(rank)
# form = "linear"
# print(f'p nDCG@{p} ({form}): {ndcg.ndcg(rel_true, rel_pred, p, form)}')

# # 240123 e.g., nphrase = 20 vs 30 vs 40 vs 50 vs 60 vs 70 vs 80 vs 90 vs 100 (C-1, p=20, cosine, exponential)
# evaluate = ["jaccard", "cosine"]
# for m in evaluate:
#     # baseline
#     df = pd.read_csv(f"/Users/smallcat/Documents/GitHub/graph/data/dnp_{m}_n100.csv", header=0)
#     col_article = df['Article'].tolist()
#     col_similarity = df['Similarity'].tolist()
#     rel_true = col_similarity

#     n_selector = list(range(20, 101, 10))
#     p_selector = [5, 10, 15, 20]

#     for n in n_selector:
#         for p in p_selector:
#             df = pd.read_csv(f"/Users/smallcat/Documents/GitHub/graph/data/dnp_{m}_n{n}.csv", header=0)
#             rank = df['Article'].tolist()[:p]
#             rel_pred =[]
#             for idx1, item1 in enumerate(rank):
#                 for idx2, item2 in enumerate(col_article):
#                     if item1 == item2:
#                         rel_pred.append(col_similarity[idx2])
#                         break
#                 else:
#                     rel_pred.append(col_similarity[-1])
#             forms = ["linear", "exp"]
#             for form in forms:
#                 print(f"{m}, {form}, n={n}, p={p}: {ndcg.ndcg(rel_true, rel_pred, p, form)}")

# 240131 e.g., pipeline size = small, medium, large (C-1, nphrase=50, noun, p=20, exponential)
evaluate = ["cosine"]
for m in evaluate:
    # baseline
    df = pd.read_csv(f"/Users/smallcat/Documents/GitHub/graph/data/dnp_{m}_n50_Large.csv", header=0)
    col_article = df['Article'].tolist()
    col_similarity = df['Similarity'].tolist()
    rel_true = col_similarity

    p_selector = [5, 10, 15, 20]
    s_selector = ["Small", "Medium", "Large"]

    for s in s_selector:
        for p in p_selector:
            df = pd.read_csv(f"/Users/smallcat/Documents/GitHub/graph/data/dnp_{m}_n50_{s}.csv", header=0)
            rank = df['Article'].tolist()[:p]
            rel_pred =[]
            for idx1, item1 in enumerate(rank):
                for idx2, item2 in enumerate(col_article):
                    if item1 == item2:
                        rel_pred.append(col_similarity[idx2])
                        break
                else:
                    rel_pred.append(col_similarity[-1])
            forms = ["linear", "exp"]
            for form in forms:
                print(f"{m}, {form}, n=50, p={p}, s={s}: {ndcg.ndcg(rel_true, rel_pred, p, form)}")