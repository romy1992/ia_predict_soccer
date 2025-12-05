import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, jaccard_score, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.model_selection import IterativeStratification

from src.service_ia.training.under_over.service.filter_uo_service import FilterUOService

X, y, key = FilterUOService(multilabel=True, list_stats=['mean_statistics']).split_dataset_single_event()
y = np.vstack(y).astype(int)

cv = IterativeStratification(n_splits=5, order=1)
base_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'  # bilancia ogni label nel singolo classificatore
)
clf = OneVsRestClassifier(base_rf, n_jobs=-1)

# 4) Metriche (subset accuracy è molto severa; usa anche F1 micro/macro e Jaccard)
scoring = {
    "subset_accuracy": make_scorer(accuracy_score),  # tutte le etichette esatte
    "f1_micro": make_scorer(f1_score, average="micro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "precision_micro": make_scorer(precision_score, average="micro", zero_division=0),
    "recall_micro": make_scorer(recall_score, average="micro", zero_division=0),
    "jaccard_samples": make_scorer(jaccard_score, average="samples", zero_division=0),
}

cv_res = cross_validate(
    clf, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

# 5) Riassunto risultati
import numpy as np

for k, vals in cv_res.items():
    if k.startswith("test_"):
        print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

label_names = ["Over1.5", "Over2.5", "Over3.5"]  # metti i tuoi nomi/ordine esatti

# Predizioni CV (0/1)
y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

# Report per label + macro/micro/weighted
print(classification_report(y, y_pred, target_names=label_names, zero_division=0))


from sklearn.metrics import multilabel_confusion_matrix

mcm = multilabel_confusion_matrix(y, y_pred)  # shape: (n_labels, 2, 2)
for j, name in enumerate(label_names):
    tn, fp, fn, tp = mcm[j].ravel()
    print(f"{name}: TP={tp} FP={fp} FN={fn} TN={tn}")


from sklearn.metrics import roc_auc_score, average_precision_score

# Probabilità CV
y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)

for j, name in enumerate(label_names):
    auc_roc = roc_auc_score(y[:, j], y_proba[:, j])
    ap      = average_precision_score(y[:, j], y_proba[:, j])  # PR-AUC
    print(f"{name}: ROC-AUC={auc_roc:.3f} | PR-AUC={ap:.3f}")

from sklearn.metrics import f1_score

thr_grid = np.linspace(0.05, 0.95, 19)
y_pred_tuned = np.zeros_like(y)
best_thr = []

for j in range(y.shape[1]):
    scores = [f1_score(y[:, j], (y_proba[:, j] >= t).astype(int)) for t in thr_grid]
    t_star = thr_grid[int(np.argmax(scores))]
    best_thr.append(t_star)
    y_pred_tuned[:, j] = (y_proba[:, j] >= t_star).astype(int)

print("Soglie ottimali per label:", dict(zip(label_names, best_thr)))
print("F1_macro tuned:", f1_score(y, y_pred_tuned, average="macro"))
print("F1_micro tuned:", f1_score(y, y_pred_tuned, average="micro"))

# (per Over: forza la gerarchia O1.5 ≥ O2.5 ≥ O3.5)
y_pred_tuned[:,1] = np.minimum(y_pred_tuned[:,1], y_pred_tuned[:,0])
y_pred_tuned[:,2] = np.minimum(y_pred_tuned[:,2], y_pred_tuned[:,1])

print("F1_macro tuned+mono:", f1_score(y, y_pred_tuned, average="macro"))
print("F1_micro tuned+mono:", f1_score(y, y_pred_tuned, average="micro"))

from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, supp = precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
for j, name in enumerate(label_names):
    print(f"{name}: P={prec[j]:.3f} R={rec[j]:.3f} F1={f1[j]:.3f} support={supp[j]}")
