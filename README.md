# Guida pratica: Parametri di bilanciamento classi con e senza SMOTE

| Modello                                             | Parametri utili **SENZA SMOTE** (dataset sbilanciato) | Parametri da **NON usare CON SMOTE** (già bilanciato) | Note pratiche                                                                 |
|-----------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------------------|
| **LogisticRegression**                              | `class_weight='balanced'`                             | `class_weight`                                        | Con SMOTE già riequilibrato, meglio lasciare neutro.                          |
| **RandomForestClassifier**                          | `class_weight='balanced'`                             | `class_weight`                                        | RF è robusto, ma il bilanciamento via weight ha senso solo se non oversampli. |
| **DecisionTreeClassifier**                          | `class_weight='balanced'`                             | `class_weight`                                        | Stesso discorso: con SMOTE diventa ridondante.                                |
| **SVC (con kernel RBF/linear)**                     | `class_weight='balanced'`                             | `class_weight`                                        | Con SMOTE meglio standardizzare le feature ma senza weight.                   |
| **KNeighborsClassifier**                            | *nessuno* (KNN non ha gestione dei pesi per classi)   | *nessuno*                                             | Usa sempre solo scaling (es. `StandardScaler`).                               |
| **XGBClassifier**                                   | `scale_pos_weight = (#neg / #pos)`                    | `scale_pos_weight`                                    | Parametro nato per dataset sbilanciati → inutile/dannoso con SMOTE.           |
| **SGDClassifier (log\_loss, hinge, ecc.)**          | `class_weight='balanced'`                             | `class_weight`                                        | Meglio settarlo solo se non applichi SMOTE.                                   |
| **Naive Bayes** (Gaussian, Complement, Multinomial) | *nessuno* (alcune varianti hanno smoothing α)         | *nessuno*                                             | Non supporta direttamente class\_weight.                                      |

# REGOLA SCALING :

Attenzione: anche se non serve, lo scaling non danneggia questi modelli. È solo ridondante e fa perdere tempo.

| Modello                                                                               | Motivo                                         |
|---------------------------------------------------------------------------------------|------------------------------------------------|
| **Decision Tree**                                                                     | Lavora per split, non su distanza              |
| **Random Forest**                                                                     | È un insieme di alberi, quindi stesso discorso |
| **Gradient Boosting** (es. `HistGradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost`) | Basato su alberi ⇒ no scaling necessario       |
| **Naive Bayes** (in particolare `CategoricalNB`, `MultinomialNB`)                     | Lavora con frequenze e conteggi, non distanze  |
| **Rule-based Models** (es. RuleFit, Explainable Boosting Machine)                     | Basati su logiche e split                      |

Modelli ad albero: DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost
Modelli rule-based: RuleFit, Explainable Boosting Machine
Naive Bayes: MultinomialNB, CategoricalNB

Modelli che richiedono scaling (scaling è fortemente consigliato o obbligatorio):

| Modello                                   | Motivo                                                                                         |
|-------------------------------------------|------------------------------------------------------------------------------------------------|
| **SVM (LinearSVC, SVC)**                  | Basato su distanze e margini                                                                   |
| **K-Nearest Neighbors (KNN)**             | Basato su distanza euclidea o simili                                                           |
| **Logistic Regression**                   | Ottimizzazione numerica ⇒ convergenza più rapida e precisa con scaling                         |
| **Linear Regression (OLS, Ridge, Lasso)** | Stessa cosa, scala influenza i coefficienti                                                    |
| **Perceptron / MLP / Reti Neurali**       | Convergenza durante l’ottimizzazione (gradient descent) migliora molto con dati standardizzati |
| **PCA, LDA**                              | Si basano su varianza, correlazione ⇒ le scale influiscono fortemente                          |

# Early Stopping: Quali modelli lo supportano e come usarlo

| Algoritmo                          | early_stopping / n_iter_no_change | eval_set richiesto | Dove si imposta? |
|------------------------------------|-----------------------------------|--------------------|------------------|
| **XGBClassifier**                  | early_stopping_rounds             | Sì                 | Nel `.fit()`     |
| **LGBMClassifier**                 | early_stopping_rounds             | Sì                 | Nel `.fit()`     |
| **HistGradientBoostingClassifier** | early_stopping, n_iter_no_change  | No                 | Nel costruttore  |
| **GradientBoostingClassifier**     | n_iter_no_change                  | No                 | Nel costruttore  |
| **RandomForestClassifier**         | No                                | No                 | —                |
| **LogisticRegression**             | No                                | No                 | —                |
| **SVC**                            | No                                | No                 | —                |
| **AdaBoostClassifier**             | No                                | No                 | —                |

**Note:**

- Per XGBoost e LightGBM, `early_stopping_rounds` e `eval_set` vanno passati nel metodo `.fit()`.
- Per i modelli scikit-learn con `n_iter_no_change` o `early_stopping`, si imposta nel costruttore.
- I modelli classici (`RandomForest`, `SVC`, `LogisticRegression`, `AdaBoost`) non supportano l’early stopping.

I modelli che supportano l’early stopping (come XGBoost, LightGBM, GradientBoosting) lo fanno perché l’addestramento
avviene in modo iterativo, aggiungendo ogni volta un nuovo “estimator” (tipicamente un albero) o passando per più
“epoche” (iterazioni).
Boosting (XGB, LGBM, GradientBoosting): aggiungono un albero alla volta, correggendo gli errori dei precedenti.
Early stopping: controlla dopo ogni iterazione/epoca se la metrica di validazione migliora. Se non migliora per un certo
numero di iterazioni, l’addestramento si ferma prima.

# Configurare stimatori base e uso di Bagging/Pasting

| Framework                    | Estimatore configurabile? | Tipo di stimatore di default          |
|------------------------------|---------------------------|---------------------------------------|
| `AdaBoostClassifier`         | ✅ `base_estimator`        | `DecisionTreeClassifier(max_depth=1)` |
| `BaggingClassifier`          | ✅ `base_estimator`        | `DecisionTreeClassifier()`            |
| `GradientBoostingClassifier` | ❌                         | `DecisionTreeRegressor(max_depth=3)`  |

# Bagging vs Pasting e modelli che li supportano

| Modello                | Usa Bagging? | Usa Pasting? | Bootstrap configurabile?        |
|------------------------|--------------|--------------|---------------------------------|
| BaggingClassifier      | ✅            | ✅            | ✅ (`bootstrap=True/False`)      |
| BaggingRegressor       | ✅            | ✅            | ✅                               |
| RandomForestClassifier | ✅ (fisso)    | ❌            | ✅ (`bootstrap=True` di default) |
| VotingClassifier       | ❌            | ❌            | ❌                               |
| GradientBoosting       | ❌            | ❌            | ❌                               |
| StackingClassifier     | ❌            | ❌            | ❌                               |

# Impostare il learning_rate nei modelli di boosting

Learning_rate: XGBoost, LightGBM e AdaBoost -> per tutti
Il learning_rate controlla quanto ogni nuovo modello contribuisce alla previsione finale.
È un moltiplicatore per l’output del nuovo albero.

predizione_finale += learning_rate * nuovo_albero(x)

| Valore `learning_rate`   | Effetto                                                  |
|--------------------------|----------------------------------------------------------|
| **Alto** (`0.5 - 1.0`)   | 💥 Veloce apprendimento ma rischio **overfitting**       |
| **Basso** (`0.01 - 0.2`) | 🛡️ Apprendimento più lento ma **più preciso e robusto** |
| **Molto basso**          | 🐢 Richiede molti più alberi (`n_estimators` alto)       |

Regola d’oro
Basso learning_rate + Alto n_estimators = Generalizzazione migliore
