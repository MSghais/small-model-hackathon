"""QA metrics and paired bootstrap significance."""

from __future__ import annotations

import random
import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(pred: str, gold: str) -> float:
    return float(normalize(gold) in normalize(pred))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def paired_bootstrap(scores_a, scores_b, iters=2000, seed=0):
    rng = random.Random(seed)
    n, wins = len(scores_a), 0
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(scores_a[i] for i in idx) / n
        db = sum(scores_b[i] for i in idx) / n
        wins += db > da
    return wins / iters
