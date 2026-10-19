"""Downstream evaluation of embeddings: SVM node classification and t-SNE."""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def create_X_y(result):
    """result has columns name, category, emb -> (X, y) with categories factorized."""
    emb_df = pd.DataFrame(data = {
        "name": result["name"],
        "category": result["category"],
        "emb": result["emb"],
    })

    emb_df['target'] = pd.factorize(emb_df['category'])[0].astype("float32")
    y = emb_df['target'].to_numpy()
    emb_df['X'] = emb_df['emb'].apply(lambda x: np.array(x))
    X = np.array(emb_df['X'].to_list())

    return X, y


def svm_accuracy(result, k_folds=5, model='linear', seed=None):
    """Repeat a 75/25 random split k_folds times and train a balanced SVC.

    seed=None keeps the original unseeded behaviour; pass an int for reproducible splits
    (split i uses random_state=seed+i).
    :return: (acc_scores, y_test, pred, classes) — the last three from the final split.
    """
    acc_scores = []

    X, y = create_X_y(result)

    for i in range(k_folds):
        random_state = None if seed is None else seed + i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
        clf = svm.SVC(kernel=model, class_weight='balanced')
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc_scores.append(accuracy_score(pred, y_test))

    return acc_scores, y_test, pred, clf.classes_


def tsne_frame(result, random_state=6):
    """2-D t-SNE coordinates with name and category columns."""
    X = np.array(list(result["emb"]))
    X_embedded = TSNE(n_components=2, random_state=random_state).fit_transform(X)

    return pd.DataFrame(data = {
        "name": result["name"],
        "category": result["category"],
        "x": [value[0] for value in X_embedded],
        "y": [value[1] for value in X_embedded],
    })


def emb_frame(u, category):
    """Embedding matrix + categories -> DataFrame with columns name, category, emb."""
    return pd.DataFrame(data = {
        "name": range(u.shape[0]),
        "category": category,
        "emb": [row for row in u],
    })
