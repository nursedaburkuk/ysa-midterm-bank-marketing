"""Ortak yardımcı kodlar: seed'ler, veri okuma, kat-içi ön işleme, PyTorch MLP,
eğitim döngüsü ve metrikler.

Bu modüldeki bütün fit işlemleri yalnızca kendisine verilen eğitim alt kümesinde
yapılır. Test verisi bu modülde hiçbir yerde otomatik olarak okunmaz.
"""
from __future__ import annotations

import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, RobustScaler,
                                   StandardScaler)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-additional.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CONFIGS = ROOT / "configs"

# ---------------------------------------------------------------------------
# 1. Seed politikası
# ---------------------------------------------------------------------------
STUDENT_LAST5 = 29048            # öğrenci numarası 244329048 -> son beş hane
BASE_SEED = STUDENT_LAST5 + 2026  # = 31074

# Alt seed'ler temel seed'e sabit bir kaydırma eklenerek deterministik biçimde
# türetilir. Aynı temel seed her zaman aynı alt seed'leri üretir.
SEED_OFFSETS = {
    "python": 0,
    "numpy": 1,
    "torch": 2,              # birincil ağırlık-başlangıç seed'i
    "train_test_split": 3,
    "outer_cv": 4,
    "inner_cv": 5,
    "early_stop_split": 6,   # iç eğitimden %15 izleme kümesini ayıran seed
    "random_search": 7,      # otomatik aday örnekleme seed'i
    "bootstrap": 8,
    "torch_stability_2": 102,  # seed kararlılığı için 2. ağırlık seed'i
    "torch_stability_3": 202,  # seed kararlılığı için 3. ağırlık seed'i
}
SEEDS = {k: BASE_SEED + v for k, v in SEED_OFFSETS.items()}


def set_global_seeds(torch_seed: int | None = None) -> None:
    """Python, NumPy ve PyTorch seed'lerini sabitler."""
    random.seed(SEEDS["python"])
    np.random.seed(SEEDS["numpy"])
    torch.manual_seed(SEEDS["torch"] if torch_seed is None else torch_seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# 2. Veri okuma ve sabit (öğrenme gerektirmeyen) kurallar
# ---------------------------------------------------------------------------
TARGET = "y"
DROP_COLS = ["duration"]  # görüşmeden önce bilinmediği için çıkarılır
NUMERIC_COLS = ["age", "campaign", "pdays", "previous", "emp.var.rate",
                "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "day_of_week", "poutcome"]
BINARY_COLS = ["pdays_no_prev_contact"]


def load_raw() -> pd.DataFrame:
    """Ham bank-additional.csv dosyasını okur (ayraç ';')."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} bulunamadı. README.md içindeki indirme adımlarını izleyin.")
    return pd.read_csv(DATA_PATH, sep=";")


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sabit kuralları uygular ve X, y döndürür.

    * y: "yes" -> 1, "no" -> 0
    * duration çıkarılır.
    * pdays = 999 "daha önce temas yok" anlamına gelir. Bu bilgi kaybolmasın
      diye ikili bir gösterge sütunu (pdays_no_prev_contact) eklenir ve pdays
      içindeki 999 değeri eksik (NaN) olarak işaretlenir. Bu NaN'lar daha sonra
      her eğitim katında, o katın fit alt kümesinden öğrenilen medyanla
      doldurulur. Satır silinmez.
    Bu işlemler satır bazlıdır, veriden parametre öğrenmez; bu yüzden sızıntı
    oluşturmaz.
    """
    df = df.copy()
    y = df[TARGET].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Hedef değişkende beklenmeyen etiket var.")
    X = df.drop(columns=[TARGET] + DROP_COLS)
    X["pdays_no_prev_contact"] = (X["pdays"] == 999).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != 999, np.nan).astype(float)
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object)
    return X, y.astype(int)


def load_split() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "dataset_split.csv")


def load_development() -> tuple[pd.DataFrame, pd.Series]:
    """Yalnızca development satırlarını döndürür."""
    X, y = prepare_xy(load_raw())
    split = load_split()
    idx = split.loc[split["subset"] == "development", "original_index"].values
    return X.loc[idx], y.loc[idx]


def load_test_features() -> pd.DataFrame:
    """Test satırlarının yalnızca girdi özelliklerini döndürür (etiket yok)."""
    X, _ = prepare_xy(load_raw())
    split = load_split()
    idx = split.loc[split["subset"] == "test", "original_index"].values
    return X.loc[idx]


def make_train_test_split(X, y):
    return train_test_split(X, y, test_size=0.20, stratify=y, shuffle=True,
                            random_state=SEEDS["train_test_split"])


# ---------------------------------------------------------------------------
# 3. Kat-içi ön işleme
# ---------------------------------------------------------------------------
class PercentileClipper(BaseEstimator, TransformerMixin):
    """Her sayısal sütun için alt/üst yüzdelik eşiklerini fit verisinden öğrenir
    ve transform sırasında değerleri bu eşiklere kırpar."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class UnknownToNaN(BaseEstimator, TransformerMixin):
    """Kategorik 'unknown' değerini NaN yapar (en sık sınıf imputation seçeneği
    için). Parametre öğrenmez."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().astype(object)
        return X.where(X != "unknown", np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler,
           "robust": RobustScaler}


def build_preprocessor(scaler: str = "standard", clipping: bool = False,
                       unknown: str = "category") -> ColumnTransformer:
    """Ön işleme hattı. Bütün parametreler fit() çağrıldığı veride öğrenilir.

    scaler   : 'standard' | 'minmax' | 'robust'
    clipping : True ise sayısal sütunlar fit verisinin %1-%99 yüzdeliklerine
               kırpılır.
    unknown  : 'category' -> 'unknown' ayrı kategori olarak kalır;
               'most_frequent' -> 'unknown' NaN sayılır ve fit verisindeki en
               sık sınıfla doldurulur.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if clipping:
        num_steps.append(("clipper", PercentileClipper(1.0, 99.0)))
    num_steps.append(("scaler", SCALERS[scaler]()))

    if unknown == "category":
        cat_steps = [("onehot", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))]
    elif unknown == "most_frequent":
        cat_steps = [("unknown_to_nan", UnknownToNaN()),
                     ("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))]
    else:
        raise ValueError(unknown)

    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_COLS),
        ("cat", Pipeline(cat_steps), CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ], remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# 4. PyTorch MLP
# ---------------------------------------------------------------------------
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Gizli katmanlar + tek logit üreten çıkış katmanı."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], activation="relu",
                 dropout: float = 0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # tek logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def n_params_formula(n_in: int, hidden) -> int:
    total, prev = 0, n_in
    for h in hidden:
        total += prev * h + h
        prev = h
    return total + prev + 1


def make_optimizer(name, params, lr, weight_decay):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    raise ValueError(name)


TRAINING = {
    "max_epochs": 150,
    "early_stopping_patience": 15,
    "restore_best_weights": True,
    "monitor_fraction": 0.15,
    "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
                  "patience": 5, "min_lr": 1e-5},
    "monitor_metric": "monitor BCE loss (mode=min)",
    "selection_metric": "inner CV average_precision_score (AP)",
    "threshold": 0.5,
    "sgd_momentum": 0.9,
}


def split_fit_monitor(X, y, seed=None):
    """İç eğitim verisinden %15'lik stratified erken-durdurma izleme kümesi
    ayırır."""
    seed = SEEDS["early_stop_split"] if seed is None else seed
    return train_test_split(X, y, test_size=TRAINING["monitor_fraction"],
                            stratify=y, shuffle=True, random_state=seed)


class FittedModel:
    """Öğrenilmiş ön işleme + eğitilmiş MLP. predict_proba yalnızca transform
    uygular."""

    def __init__(self, preprocessor, model, config, history, info):
        self.preprocessor = preprocessor
        self.model = model
        self.config = config
        self.history = history
        self.info = info

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model.eval()
        Z = torch.tensor(self.preprocessor.transform(X), dtype=torch.float32)
        return torch.sigmoid(self.model(Z)).numpy()


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series, config: dict,
              weight_seed: int | None = None) -> FittedModel:
    """Bir konfigürasyonu verilen eğitim verisinde eğitir.

    1) X_train -> fit (%85) + monitor (%15), stratified.
    2) Ön işleme yalnızca fit alt kümesinde öğrenilir; monitor'a transform.
    3) MLP fit alt kümesinde eğitilir; her epoch sonunda monitor BCE kaybı
       hesaplanır; scheduler (açıksa) ve erken durdurma bu kayıpla çalışır.
    4) En iyi monitor kaybına ait ağırlıklar geri yüklenir.
    """
    weight_seed = SEEDS["torch"] if weight_seed is None else weight_seed
    t0 = time.perf_counter()
    X_fit, X_mon, y_fit, y_mon = split_fit_monitor(X_train, y_train)

    pre = build_preprocessor(config["scaler"], config.get("clipping", False),
                             config.get("unknown", "category"))
    Z_fit = pre.fit_transform(X_fit)
    Z_mon = pre.transform(X_mon)

    torch.manual_seed(weight_seed)
    model = MLP(Z_fit.shape[1], tuple(config["hidden_layers"]),
                config.get("activation", "relu"), config.get("dropout", 0.0))
    opt = make_optimizer(config["optimizer"], model.parameters(),
                         config["lr"], config["weight_decay"])
    sch_cfg = TRAINING["scheduler"]
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=sch_cfg["factor"], patience=sch_cfg["patience"],
        min_lr=sch_cfg["min_lr"]) if config.get("scheduler", True) else None)
    loss_fn = nn.BCEWithLogitsLoss()

    Xf = torch.tensor(Z_fit, dtype=torch.float32)
    yf = torch.tensor(y_fit.values, dtype=torch.float32)
    Xm = torch.tensor(Z_mon, dtype=torch.float32)
    ym = torch.tensor(y_mon.values, dtype=torch.float32)
    gen = torch.Generator().manual_seed(weight_seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xf, yf),
        batch_size=int(config["batch_size"]), shuffle=True, generator=gen)

    best_loss, best_epoch, best_state, wait = np.inf, 0, None, 0
    history, warn_msgs = [], []
    stopped_epoch = TRAINING["max_epochs"]
    for epoch in range(1, TRAINING["max_epochs"] + 1):
        lr_now = opt.param_groups[0]["lr"]
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        train_loss = tot / n
        model.eval()
        with torch.no_grad():
            mon_loss = loss_fn(model(Xm), ym).item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "monitor_loss": mon_loss, "lr": lr_now})
        if not np.isfinite(train_loss) or not np.isfinite(mon_loss):
            warn_msgs.append(f"epoch {epoch}: sonlu olmayan kayıp")
            stopped_epoch = epoch
            break
        if scheduler is not None:
            scheduler.step(mon_loss)
        if mon_loss < best_loss:
            best_loss, best_epoch, wait = mon_loss, epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= TRAINING["early_stopping_patience"]:
                stopped_epoch = epoch
                break
    early_stopped = stopped_epoch < TRAINING["max_epochs"] or (
        wait >= TRAINING["early_stopping_patience"])
    if not early_stopped:
        warn_msgs.append("150 epoch üst sınırına ulaşıldı; erken durdurma "
                         "tetiklenmedi (yakınsama tamamlanmamış olabilir)")
    if best_state is not None:
        model.load_state_dict(best_state)  # en iyi ağırlıkları geri yükle
    info = {
        "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
        "best_monitor_loss": best_loss, "early_stopped": bool(early_stopped),
        "final_lr": opt.param_groups[0]["lr"],
        "train_time_s": time.perf_counter() - t0,
        "n_fit": len(y_fit), "n_monitor": len(y_mon),
        "n_inputs": int(Z_fit.shape[1]), "n_params": count_parameters(model),
        "warnings": " | ".join(warn_msgs),
    }
    return FittedModel(pre, model, config, pd.DataFrame(history), info)


# ---------------------------------------------------------------------------
# 5. Metrikler
# ---------------------------------------------------------------------------
METRIC_NAMES = ["AP", "ROC_AUC", "Accuracy", "Precision", "Sensitivity",
                "Specificity", "F1", "Balanced_Accuracy"]


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AP": average_precision_score(y_true, prob),
        "ROC_AUC": roc_auc_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def mean_sd(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
