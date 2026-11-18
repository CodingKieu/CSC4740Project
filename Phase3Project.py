"""
Group project by Khai Kieu
Phase 3: content-based model, adaptive hybrid, ablation and ranking metrics.
Reuses the data pipeline, split and baselines from Phase2Project.py.
Run: python Phase3Project.py [path/to/folder/with/ratings.dat]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import Phase2Project as p2

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
MF_FACTORS = [20, 50]
MF_LAMBDAS = [10, 20, 30]
MF_ITERS = 12
CB_ALPHAS = [1, 5, 20, 50]
K_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]
FIXED_GRID = np.round(np.arange(0.0, 1.01, 0.1), 1)
TOP_N = 10


def load_genres(data_dir: Path, ratings: pd.DataFrame) -> np.ndarray:
    """multi-hot genre matrix indexed by the contiguous item index i."""
    movies = pd.read_csv(data_dir / "movies.dat", sep="::", engine="python",
                         names=["movie_id", "title", "genres"], encoding="latin-1")
    first = ratings.drop_duplicates("i").sort_values("i")
    genres = movies.set_index("movie_id")["genres"].reindex(first["movie_id"].to_numpy())
    mat = np.zeros((len(first), len(GENRES)))
    for row, g in enumerate(genres.fillna("")):
        for name in g.split("|"):
            if name in GENRES:
                mat[row, GENRES.index(name)] = 1.0
    return mat


def als_mf(train, shape, n_factors, lam, n_iter=MF_ITERS):
    """biased mf: mu + b_u + b_i + p_u.q_i, fitted by ALS on observed ratings only."""
    rng = np.random.default_rng(p2.SEED)
    mu = train["rating"].mean()
    by_user = csr_matrix((train["rating"] - mu, (train["u"], train["i"])), shape=shape)
    by_item = by_user.T.tocsr()
    P = 0.1 * rng.standard_normal((shape[0], n_factors))
    Q = 0.1 * rng.standard_normal((shape[1], n_factors))
    b_u, b_i = np.zeros(shape[0]), np.zeros(shape[1])
    reg = lam * np.eye(n_factors + 1)

    def solve_side(mat, own_b, own_f, other_b, other_f):
        for r in range(mat.shape[0]):
            s, e = mat.indptr[r], mat.indptr[r + 1]
            if s == e:
                continue
            idx = mat.indices[s:e]
            X = np.hstack([np.ones((e - s, 1)), other_f[idx]])
            y = mat.data[s:e] - other_b[idx]
            sol = np.linalg.solve(X.T @ X + reg, X.T @ y)
            own_b[r], own_f[r] = sol[0], sol[1:]

    for _ in range(n_iter):
        solve_side(by_user, b_u, P, b_i, Q)
        solve_side(by_item, b_i, Q, b_u, P)
    return mu, b_u, b_i, P, Q


def mf_predict(model, df):
    mu, b_u, b_i, P, Q = model
    u, i = df["u"].to_numpy(), df["i"].to_numpy()
    return np.clip(mu + b_u[u] + b_i[i] + np.einsum("ij,ij->i", P[u], Q[i]), 1, 5)


def cb_predictions(train, dfs, genre_mat, alpha):
    """per-user ridge on genre vectors, target = rating - item mean (train only)."""
    gm = train["rating"].mean()
    item_mean = train.groupby("i")["rating"].mean().reindex(range(len(genre_mat))).fillna(gm)
    item_mean = item_mean.to_numpy()
    tr = train.reset_index(drop=True)
    resid = tr["rating"].to_numpy() - item_mean[tr["i"].to_numpy()]
    items = tr["i"].to_numpy()
    n_users = tr["u"].max() + 1
    W = np.zeros((n_users, genre_mat.shape[1]))
    c = np.zeros(n_users)
    for u, idx in tr.groupby("u").indices.items():
        G = genre_mat[items[idx]]
        gc, rc = G - G.mean(0), resid[idx] - resid[idx].mean()
        w = np.linalg.solve(gc.T @ gc + alpha * np.eye(G.shape[1]), gc.T @ rc)
        W[u], c[u] = w, resid[idx].mean() - G.mean(0) @ w  # unpenalised intercept
    return [np.clip(item_mean[d["i"]] + c[d["u"]]
                    + np.einsum("ij,ij->i", W[d["u"]], genre_mat[d["i"]]), 1, 5)
            for d in dfs]


def blend(p_cf, p_cb, alpha):
    return alpha * p_cf + (1 - alpha) * p_cb


def ranking_metrics(test, preds, groups):
    """precision@N and ndcg@N over each user's held-out items (relevant = rating >= 4)."""
    df = pd.DataFrame({"u": test["u"].to_numpy(), "rel": (test["rating"] >= 4).to_numpy(),
                       "pred": preds, "g": groups.to_numpy()})
    rows = []
    disc = 1 / np.log2(np.arange(2, TOP_N + 2))
    for u, d in df.groupby("u"):
        n = min(TOP_N, len(d))
        rel = d.sort_values("pred", ascending=False)["rel"].to_numpy()[:n].astype(float)
        ideal = np.sort(d["rel"].to_numpy().astype(float))[::-1][:n]
        idcg = (ideal * disc[:n]).sum()
        rows.append((d["g"].iloc[0], rel.mean(), (rel * disc[:n]).sum() / idcg if idcg else 0.0))
    out = pd.DataFrame(rows, columns=["g", "p", "n"])
    res = {f"P@{TOP_N}": out["p"].mean(), f"NDCG@{TOP_N}": out["n"].mean()}
    for lab in p2.GROUP_LABELS:
        res[f"NDCG {lab}"] = out.loc[out["g"] == lab, "n"].mean()
    return res


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else p2.DATA_DIR
    p2.RESULTS_DIR.mkdir(exist_ok=True)
    ratings = p2.load_and_clean(data_dir)
    shape = (ratings["u"].nunique(), ratings["i"].nunique())
    splits = p2.split_per_user(ratings)
    train, val, test = splits["train"], splits["val"], splits["test"]
    genre_mat = load_genres(data_dir, ratings)
    n_train = train.groupby("u").size()
    groups = pd.cut(test["u"].map(n_train), bins=p2.GROUP_BINS, labels=p2.GROUP_LABELS)
    val_y = val["rating"].to_numpy()

    # 1. collaborative component: tune MF on validation only
    mf_val = {}
    for f in MF_FACTORS:
        for lam in MF_LAMBDAS:
            m = als_mf(train, shape, f, lam)
            mf_val[(f, lam)] = (p2.rmse(val_y, mf_predict(m, val)), m)
            print(f"MF f={f} lambda={lam}: val rmse {mf_val[(f, lam)][0]:.4f}", flush=True)
    best_mf = min(mf_val, key=lambda k: mf_val[k][0])
    mf_model = mf_val[best_mf][1]
    mf_v, mf_t = mf_predict(mf_model, val), mf_predict(mf_model, test)

    # 2. content-based component: tune ridge alpha on validation only
    cb_val = {a: p2.rmse(val_y, cb_predictions(train, [val], genre_mat, a)[0]) for a in CB_ALPHAS}
    best_a = min(cb_val, key=cb_val.get)
    print("CB val rmse:", {a: round(v, 4) for a, v in cb_val.items()}, "-> best", best_a)
    cb_v, cb_t = cb_predictions(train, [val, test], genre_mat, best_a)

    # 3. item knn (collaborative alternative)
    knn_v = p2.knn_predictions(train, val, shape, p2.N_NEIGHBORS)
    knn_t = p2.knn_predictions(train, test, shape, p2.N_NEIGHBORS)

    # 4. tune k (adaptive) and alpha (fixed) on validation only
    def tune(p_cf_v):
        n_v = val["u"].map(n_train).to_numpy()
        ak = {k: p2.rmse(val_y, blend(p_cf_v, cb_v, n_v / (n_v + k))) for k in K_GRID}
        af = {a: p2.rmse(val_y, blend(p_cf_v, cb_v, a)) for a in FIXED_GRID}
        return min(ak, key=ak.get), ak, min(af, key=af.get), af
    k_mf, ak_mf, a_mf, af_mf = tune(mf_v)
    k_knn, ak_knn, a_knn, af_knn = tune(knn_v)
    print("adaptive k (MF):", {k: round(v, 4) for k, v in ak_mf.items()}, "-> best", k_mf)
    print("fixed alpha (MF):", {float(a): round(v, 4) for a, v in af_mf.items()}, "-> best", a_mf)
    print("adaptive k (kNN):", {k: round(v, 4) for k, v in ak_knn.items()}, "-> best", k_knn)
    print("fixed alpha (kNN):", {float(a): round(v, 4) for a, v in af_knn.items()}, "-> best", a_knn)
    pd.DataFrame({"k": K_GRID, "val_rmse_mf": [ak_mf[k] for k in K_GRID],
                  "val_rmse_knn": [ak_knn[k] for k in K_GRID]}).round(4).to_csv(
        p2.RESULTS_DIR / "phase3_k_tuning.csv", index=False)

    # 5. test evaluation
    n_t = test["u"].map(n_train).to_numpy()
    models = {
        "global mean": np.full(len(test), train["rating"].mean()),
        "user mean": p2.mean_predictions(train, test, "u"),
        "item mean": p2.mean_predictions(train, test, "i"),
        f"item knn (k={p2.N_NEIGHBORS})": knn_t,
        f"biased mf (f={best_mf[0]}, lam={best_mf[1]})": mf_t,
        f"content-based (ridge={best_a})": cb_t,
        f"fixed hybrid mf+cb (a={a_mf})": blend(mf_t, cb_t, a_mf),
        f"adaptive hybrid mf+cb (k={k_mf})": blend(mf_t, cb_t, n_t / (n_t + k_mf)),
        f"fixed hybrid knn+cb (a={a_knn})": blend(knn_t, cb_t, a_knn),
        f"adaptive hybrid knn+cb (k={k_knn})": blend(knn_t, cb_t, n_t / (n_t + k_knn)),
    }
    rows = []
    for name, pr in models.items():
        row = p2.evaluate(name, test, pr, groups)
        row.update(ranking_metrics(test, pr, groups))
        rows.append(row)
    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    table.to_csv(p2.RESULTS_DIR / "phase3_results.csv", index=False)
    print(groups.value_counts().sort_index().to_string())
    print("avg adaptive alpha by group (MF):",
          pd.Series(n_t / (n_t + k_mf)).groupby(groups.to_numpy()).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
