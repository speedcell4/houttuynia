from typing import Tuple

import torch


def classification_metrics(outputs, targets, alpha: float = 0.5) -> Tuple[float, float, float]:
    L = max(outputs.max(), targets.max()).item() + 1
    matrix = torch.zeros(L, L).float()
    matrix[outputs, targets] += 1.
    true = torch.diag(matrix)
    acc = matrix.sum(0)
    rec = matrix.sum(1)
    print(true)
    print(acc)
    print(rec)
    return true / acc, true / rec, rec * acc / (alpha * rec + (1 - alpha) * acc)


def accuracy_metric(outputs, targets) -> float:
    acc, _, _ = classification_metrics(outputs, targets)
    return acc


def recall_metric(outputs, targets):
    _, rec, _ = classification_metrics(outputs, targets)
    return rec


def f_metric(outputs, targets, alpha: float = 0.5):
    _, _, f1s = classification_metrics(outputs, targets, alpha=alpha)
    return f1s


a = torch.tensor([0, 1, 2], dtype=torch.int64)
b = torch.tensor([0, 0, 2], dtype=torch.int64)

print(classification_metrics(a, b))
