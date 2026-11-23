"""Manuel (M1-M6) ve otomatik (R1-R6) adayları, arama uzayını, eğitim ayarlarını
ve seed'leri configs/ klasörüne yazar.

Bu betik nested CV başlamadan önce bir kez çalıştırılır ve çıktıları ayrı bir
commit ile kaydedilir. Hiçbir model sonucu kullanmaz.

Çalıştırma: python -m src.make_configs
"""
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

from src import project as P

# --- Manuel adaylar --------------------------------------------------------
COMMON = {"batch_size": 64, "dropout": 0.0, "activation": "relu",
          "weight_decay": 1e-4, "clipping": False, "unknown": "category"}
M1 = {**COMMON, "optimizer": "adamw", "hidden_layers": [32],
      "scaler": "standard", "lr": 1e-3, "scheduler": True}
MANUAL = {
    "M1": {**M1, "purpose": "Temel konfigürasyon"},
    "M2": {**M1, "optimizer": "adam", "purpose": "Optimizer etkisi (AdamW -> Adam)"},
    "M3": {**M1, "optimizer": "sgd", "purpose": "Optimizer etkisi (AdamW -> SGD+Momentum 0.9)"},
    "M4": {**M1, "scaler": "minmax", "purpose": "Ölçekleme etkisi (Standard -> MinMax)"},
    "M5": {**M1, "hidden_layers": [64, 32], "purpose": "Mimari etkisi ((32,) -> (64,32))"},
    "M6": {**M1, "scheduler": False, "purpose": "Learning-rate-drop etkisi (scheduler kapalı)"},
}

# --- Otomatik arama uzayı --------------------------------------------------
SEARCH_SPACE = {
    "optimizer": ["sgd", "adam", "adamw"],
    "candidates_per_optimizer": 2,
    "hidden_layers": [[16], [32], [64], [64, 32], [128, 64]],
    "activation": ["relu", "tanh"],
    "lr": {"adam": [1e-4, 1e-2], "adamw": [1e-4, 1e-2], "sgd": [1e-3, 1e-1],
           "distribution": "log-uniform"},
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "scaler": ["standard", "minmax", "robust"],
    "dropout": [0.0, 0.2],
    "clipping": [False, True],
    "fixed": {"scheduler": True, "unknown": "category",
              "sgd_momentum": 0.9},
    "sampler": "sklearn.model_selection.ParameterSampler",
    "random_state_rule": "SEEDS['random_search'] + optimizer_index "
                         "(sgd=0, adam=1, adamw=2)",
}


def sample_random_candidates():
    out, k = {}, 1
    for i, opt in enumerate(SEARCH_SPACE["optimizer"]):
        lo, hi = SEARCH_SPACE["lr"][opt]
        dist = {
            "hidden_layers": SEARCH_SPACE["hidden_layers"],
            "activation": SEARCH_SPACE["activation"],
            "lr": loguniform(lo, hi),
            "weight_decay": SEARCH_SPACE["weight_decay"],
            "batch_size": SEARCH_SPACE["batch_size"],
            "scaler": SEARCH_SPACE["scaler"],
            "dropout": SEARCH_SPACE["dropout"],
            "clipping": SEARCH_SPACE["clipping"],
        }
        sampler = ParameterSampler(dist, n_iter=2,
                                   random_state=P.SEEDS["random_search"] + i)
        for params in sampler:
            cfg = {"optimizer": opt, **params, "scheduler": True,
                   "unknown": "category"}
            cfg["lr"] = float(round(cfg["lr"], 8))
            cfg["weight_decay"] = float(cfg["weight_decay"])
            cfg["batch_size"] = int(cfg["batch_size"])
            cfg["dropout"] = float(cfg["dropout"])
            cfg["clipping"] = bool(cfg["clipping"])
            out[f"R{k}"] = cfg
            k += 1
    return out


SELECTION_RULE = {
    "within_search": "İç CV ortalama AP en yüksek aday seçilir. Eşitlikte sırasıyla "
                     "daha düşük iç CV SD, daha az öğrenilebilir parametre ve "
                     "daha küçük aday kodu (M1<M2..., R1<R2...) seçilir.",
    "final_method": "Tüm development üzerinde tekrarlanan iç CV'de manuel kazanan "
                    "ile otomatik kazanandan ortalama AP'si yüksek olan nihai "
                    "konfigürasyondur. Eşitlikte aynı ikincil kurallar uygulanır.",
    "primary_seed": "SEEDS['torch'] ile eğitilen model ana modeldir.",
    "threshold": 0.5,
}

REFERENCE_MODELS = {
    "Dummy": {"class": "DummyClassifier", "strategy": "most_frequent"},
    "LogReg": {"class": "LogisticRegression", "C": 1.0, "solver": "lbfgs",
               "max_iter": 5000,
               "preprocessing": "M1 ile aynı sabit kural: median imputation, "
                                "StandardScaler, clipping yok, 'unknown' ayrı "
                                "kategori, OneHotEncoder(handle_unknown='ignore')"},
}

SUPPLEMENTARY = {
    "note": "Ek analizler seçim bütçesine dahil değildir ve hiçbir seçimi "
            "etkilemez. Yalnızca tüm development üzerindeki iç CV katlarında, "
            "M1'den tek bir ön işleme faktörü değiştirilerek çalıştırılır.",
    "S_robust": {**M1, "scaler": "robust"},
    "S_clip": {**M1, "clipping": True},
    "S_unknown_mf": {**M1, "unknown": "most_frequent"},
}

if __name__ == "__main__":
    P.CONFIGS.mkdir(exist_ok=True)
    P.save_json(MANUAL, P.CONFIGS / "manual_configs.json")
    P.save_json(sample_random_candidates(), P.CONFIGS / "random_configs.json")
    P.save_json(SEARCH_SPACE, P.CONFIGS / "search_space.json")
    P.save_json({**P.TRAINING, "selection_rule": SELECTION_RULE,
                 "reference_models": REFERENCE_MODELS,
                 "outer_cv": {"n_splits": 3, "shuffle": True,
                              "random_state": P.SEEDS["outer_cv"]},
                 "inner_cv": {"n_splits": 3, "shuffle": True,
                              "random_state": P.SEEDS["inner_cv"]}},
                P.CONFIGS / "training_settings.json")
    P.save_json(SUPPLEMENTARY, P.CONFIGS / "supplementary_configs.json")
    P.save_json({"student_number_last5": P.STUDENT_LAST5,
                 "base_seed": P.BASE_SEED, "offsets": P.SEED_OFFSETS,
                 "seeds": P.SEEDS}, P.CONFIGS / "seeds.json")
    print("configs/ yazıldı")
