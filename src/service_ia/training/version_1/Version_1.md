Nella versione 1.0 ci sarà un processo di addestramento per trovare i migliori
modelli per prevedere eventi di under/over 1.5,2.5 e 3.5 NON COERENTI (inconsistent), ovvero
quelli eventi che non vengono addestrati con le previsioni dell'altro modello per
aumentare la probabilità, per esempio: over 1.5 non verrà dato a over 2.5 o 3.5 ma verranno
addestrati e previsti distintamente.


| Modello                                             | Parametri utili **SENZA SMOTE** (dataset sbilanciato) | Parametri da **NON usare CON SMOTE** (già bilanciato) | Note pratiche                                                                 |
| --------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| **LogisticRegression**                              | `class_weight='balanced'`                             | `class_weight`                                        | Con SMOTE già riequilibrato, meglio lasciare neutro.                          |
| **RandomForestClassifier**                          | `class_weight='balanced'`                             | `class_weight`                                        | RF è robusto, ma il bilanciamento via weight ha senso solo se non oversampli. |
| **DecisionTreeClassifier**                          | `class_weight='balanced'`                             | `class_weight`                                        | Stesso discorso: con SMOTE diventa ridondante.                                |
| **SVC (con kernel RBF/linear)**                     | `class_weight='balanced'`                             | `class_weight`                                        | Con SMOTE meglio standardizzare le feature ma senza weight.                   |
| **KNeighborsClassifier**                            | *nessuno* (KNN non ha gestione dei pesi per classi)   | *nessuno*                                             | Usa sempre solo scaling (es. `StandardScaler`).                               |
| **XGBClassifier**                                   | `scale_pos_weight = (#neg / #pos)`                    | `scale_pos_weight`                                    | Parametro nato per dataset sbilanciati → inutile/dannoso con SMOTE.           |
| **SGDClassifier (log\_loss, hinge, ecc.)**          | `class_weight='balanced'`                             | `class_weight`                                        | Meglio settarlo solo se non applichi SMOTE.                                   |
| **Naive Bayes** (Gaussian, Complement, Multinomial) | *nessuno* (alcune varianti hanno smoothing α)         | *nessuno*                                             | Non supporta direttamente class\_weight.                                      |
