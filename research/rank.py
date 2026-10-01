"""
Sentence ranking in plain numpy: TF-IDF vectors, cosine similarity, TextRank
centrality, maximal marginal relevance (MMR) and a small k-means.

Sizes are capped (vocabulary, float32) so a few thousand sentences stay within a few
tens of MB - fine on a 4 GB machine.
"""

from collections import Counter

import numpy as np


def tfidf_matrix(docs, max_vocab=5000):
    """
    docs: list of token lists. Returns (X, vocab, idf): X is (n_docs, n_terms) float32
    with L2-normalised rows (all-zero rows stay zero), vocab maps term -> column.
    Terms are the max_vocab most frequent by document frequency.
    """
    df = Counter()                                # document frequency: sentences containing the term
    for toks in docs:
        df.update(set(toks))
    terms = [t for t, _ in df.most_common(max_vocab)]
    vocab = {t: i for i, t in enumerate(terms)}
    n = len(docs)
    # smoothed idf (as in scikit-learn): rare terms weigh more, +1 keeps terms in every doc above 0
    idf = np.array([np.log((1 + n) / (1 + df[t])) + 1.0 for t in terms], dtype=np.float32)
    X = np.zeros((n, len(terms)), dtype=np.float32)
    for i, toks in enumerate(docs):
        for t, c in Counter(toks).items():
            j = vocab.get(t)
            if j is not None:
                X[i, j] = (1.0 + np.log(c)) * idf[j]     # sublinear tf: a word said 3 times isn't 3x the topic
    return _normalise_rows(X), vocab, idf          # unit rows, so X @ X.T is cosine similarity


def transform(tokens, vocab, idf):
    """TF-IDF vector (L2-normalised) of one token list in an existing vocabulary."""
    v = np.zeros(len(vocab), dtype=np.float32)
    for t, c in Counter(tokens).items():
        j = vocab.get(t)
        if j is not None:
            v[j] = (1.0 + np.log(c)) * idf[j]
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _normalise_rows(X):
    """X with each row scaled to unit length (all-zero rows left as they are)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def cosine_sim(X):
    """Pairwise cosine similarity of L2-normalised rows."""
    return X @ X.T


def textrank(S, damping=0.85, threshold=0.1, iters=30):
    """
    Centrality of each sentence in the similarity graph S (PageRank over edges with
    similarity above `threshold`). Returns scores summing to 1.
    """
    n = S.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    # weighted graph: weak links (a single common word shared) dropped, no self-loops
    W = np.where(S > threshold, S, 0.0).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    out = W.sum(axis=1, keepdims=True)
    dangling = out[:, 0] == 0                     # sentences linked to nothing
    out[dangling] = 1.0                           # avoid 0/0; their rows of P stay zero
    P = W / out                                   # row-stochastic (dangling rows all zero)
    r = np.full(n, 1.0 / n, dtype=np.float32)
    # power iteration: a random walker follows an edge (p = damping) or jumps anywhere;
    # 30 steps is plenty since the error shrinks by `damping` each step (0.85^30 < 1%)
    for _ in range(iters):
        leak = r[dangling].sum() / n               # dangling nodes spread evenly
        r = (1 - damping) / n + damping * (r @ P + leak)
    return r / r.sum()


def mmr_pick(relevance, S, pool, chosen, lam=0.7):
    """
    The best next item from `pool` by maximal marginal relevance:
    lam * relevance - (1 - lam) * max similarity to anything already `chosen`.
    Returns an index from pool, or None if pool is empty.
    """
    if not pool:
        return None
    pool = np.asarray(pool)
    rel = relevance[pool]
    red = S[np.ix_(pool, chosen)].max(axis=1) if chosen else np.zeros(len(pool))
    return int(pool[int(np.argmax(lam * rel - (1 - lam) * red))])


def kmeans(X, k, rng, iters=20):
    """Plain k-means on rows of X (k-means++ seeding). Returns (labels, centroids)."""
    n = X.shape[0]
    k = max(1, min(k, n))
    sq = (X * X).sum(axis=1)

    def dist2(C):                                 # (n, k) squared distances without an n*k*d array
        """|x - c|^2 = |x|^2 - 2 x.c + |c|^2 for every row x of X and centre c of C."""
        return np.maximum(sq[:, None] - 2 * X @ C.T + (C * C).sum(axis=1)[None, :], 0)   # clip rounding < 0

    # k-means++ seeding: first centre at random, each next one drawn with probability
    # proportional to its squared distance from the nearest centre so far - spread-out
    # starts, so clusters rarely collapse onto one theme
    centres = [X[int(rng.integers(n))]]
    for _ in range(1, k):
        d = dist2(np.array(centres)).min(axis=1)
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)   # all points on a centre: uniform
        centres.append(X[int(rng.choice(n, p=p))])
    C = np.array(centres)
    labels = np.zeros(n, dtype=int)
    # Lloyd iterations: assign to the nearest centre, move each centre to its members' mean
    for _ in range(iters):
        labels = np.argmin(dist2(C), axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members):                      # an empty cluster keeps its old centre
                C[j] = members.mean(axis=0)
    return labels, C
