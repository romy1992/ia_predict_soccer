import numpy as np
from scipy.stats import poisson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.service_ia.training.under_over.service.filter_uo_service import FilterUOService


class OverUnderFromLambda(BaseEstimator, TransformerMixin):
    def __init__(self, thresholds=(1.5, 2.5, 3.5)):
        self.thresholds = thresholds

    def fit(self, X, y=None):  # y = gol totali reali in training
        return self

    def transform(self, lam):
        # lam shape: (n_samples,) oppure (n_samples, 1)
        lam = np.asarray(lam).reshape(-1)
        probs_over = []
        for th in self.thresholds:
            k = int(np.floor(th))  # 1.5->1, 2.5->2, 3.5->3
            p_over = 1 - poisson.cdf(k, lam)
            probs_over.append(p_over)
        # enforce monotonicity (non serve, già monotone, ma per sicurezza)
        p15, p25, p35 = probs_over
        p25 = np.minimum(p25, p15)
        p35 = np.minimum(p35, p25)
        return np.vstack([p15, p25, p35]).T


def pick_labels(probs, thresholds=(0.5, 0.5, 0.5)):
    # assegna label coerenti con soglie indipendenti
    p15, p25, p35 = probs.T
    y15 = (p15 >= thresholds[0]).astype(int)
    y25 = (np.minimum(p25, p15) >= thresholds[1]).astype(int)
    y35 = (np.minimum(p35, p25) >= thresholds[2]).astype(int)
    return np.vstack([y15, y25, y35]).T


# Esempio pipeline
poisson_pipeline = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  # o with_mean=True se densi
    ("model", PoissonRegressor(alpha=1.0, max_iter=1000)),
])

service = FilterUOService(list_stats=['mean_statistics', 'shots'])
dataset = service.get_match_total_goals()
df = service.split_dataset_total_goals(dataset=dataset)
X = df.drop(columns=['id_fixture', 'y', 'name_home', 'name_away']).values
y = df['y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
poisson_pipeline.fit(X_train, y_train)
lam_pred = poisson_pipeline.predict(X_test)  # λ
probs = OverUnderFromLambda().transform(lam_pred)  # P(over1.5/2.5/3.5)
y_pred = pick_labels(probs)
print("Predizioni Over/Under 1.5, 2.5, 3.5:")
print(y_pred)

# etichette vere multilabel derivate dai gol reali
Y_true = np.column_stack([
    (y_test > 1.5).astype(int),
    (y_test > 2.5).astype(int),
    (y_test > 3.5).astype(int),
])

# metriche multilabel
subset_acc  = accuracy_score(Y_true, y_pred)                     # esattezza su tutte e 3 insieme
f1_micro    = f1_score(Y_true, y_pred, average='micro')
f1_macro    = f1_score(Y_true, y_pred, average='macro')
jaccard     = jaccard_score(Y_true, y_pred, average='samples')   # simile a "subset accuracy" più morbida
report      = classification_report(Y_true, y_pred, target_names=['over1.5','over2.5','over3.5'])

print(f"Subset accuracy: {subset_acc:.4f}")
print(f"F1 micro:        {f1_micro:.4f}")
print(f"F1 macro:        {f1_macro:.4f}")
print(f"Jaccard samples: {jaccard:.4f}")
print(report)

# AUC per ciascuna soglia (usando le probabilità monotone in 'probs')
auc_per_label = roc_auc_score(Y_true, probs, average=None)
print("ROC AUC per soglia [1.5, 2.5, 3.5]:", np.round(auc_per_label, 4))
