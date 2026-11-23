"""Nested CV, iç CV araması ve referans modeller."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import project as P


def outer_cv():
    return StratifiedKFold(n_splits=3, shuffle=True,
                           random_state=P.SEEDS["outer_cv"])


def inner_cv():
    return StratifiedKFold(n_splits=3, shuffle=True,
                           random_state=P.SEEDS["inner_cv"])


def run_inner_search(X, y, candidates: dict, splits, context: dict,
                     histories: list) -> pd.DataFrame:
    """Her aday için verilen iç katlarda eğitim yapar ve iç doğrulama AP'sini
    hesaplar. Ön işleme her eğitimde yalnızca o eğitimin fit alt kümesinde
    öğrenilir."""
    rows = []
    for code, cfg in candidates.items():
        cfg = {k: v for k, v in cfg.items() if k != "purpose"}
        for j, (tr, va) in enumerate(splits, start=1):
            m = P.train_mlp(X.iloc[tr], y.iloc[tr], cfg)
            prob = m.predict_proba(X.iloc[va])
            ap = average_precision_score(y.iloc[va], prob)
            rows.append({**context, "candidate": code, "inner_fold": j,
                         "n_inner_train": len(tr), "n_inner_val": len(va),
                         "AP": ap, **m.info})
            h = m.history.copy()
            for k, v in {**context, "candidate": code, "inner_fold": j}.items():
                h[k] = v
            histories.append(h)
    return pd.DataFrame(rows)


def summarize_search(runs: pd.DataFrame, candidates: dict) -> pd.DataFrame:
    g = runs.groupby("candidate")
    s = pd.DataFrame({
        "mean_AP": g["AP"].mean(), "sd_AP": g["AP"].std(ddof=1),
        "fold_APs": g["AP"].apply(lambda v: ";".join(f"{a:.4f}" for a in v)),
        "mean_best_epoch": g["best_epoch"].mean(),
        "mean_stopped_epoch": g["stopped_epoch"].mean(),
        "n_hit_150": g["early_stopped"].apply(lambda v: int((~v).sum())),
        "mean_time_s": g["train_time_s"].mean(),
        "total_time_s": g["train_time_s"].sum(),
        "n_params": g["n_params"].first(),
        "n_warnings": g["warnings"].apply(lambda v: int((v != "").sum())),
    })
    s = s.loc[list(candidates.keys())]
    for key in ["optimizer", "hidden_layers", "activation", "lr", "weight_decay",
                "batch_size", "scaler", "dropout", "clipping", "scheduler"]:
        s[key] = [str(candidates[c].get(key)) for c in s.index]
    return s


def select_candidate(summary: pd.DataFrame) -> str:
    """Önceden tanımlı kural: en yüksek ortalama AP; eşitlikte düşük SD, az
    parametre, küçük aday kodu."""
    s = summary.copy()
    s["code_order"] = [int(c[1:]) for c in s.index]
    s = s.sort_values(["mean_AP", "sd_AP", "n_params", "code_order"],
                      ascending=[False, True, True, True])
    return s.index[0]


def make_logreg():
    return Pipeline([("pre", P.build_preprocessor("standard", False, "category")),
                     ("clf", LogisticRegression(C=1.0, solver="lbfgs",
                                                max_iter=5000))])


def make_dummy():
    return DummyClassifier(strategy="most_frequent")


def fit_reference(name, X_tr, y_tr):
    t0 = time.perf_counter()
    model = make_logreg() if name == "LogReg" else make_dummy()
    model.fit(X_tr, y_tr)
    return model, time.perf_counter() - t0
