import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV


class CheckDataset:

    @staticmethod
    def check_model(**kwargs):
        X = kwargs.get("X")
        y = kwargs.get("y")

        # hyperparametric base
        degree = kwargs.get("degree", 1)
        alpha = kwargs.get("alpha", 1.0)

        # train/validation split
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2, random_state=42)

        # se viene passato un param_grid, eseguo GridSearchCV
        param_grid = kwargs.get("param_grid")
        use_grid_search = param_grid is not None

        if use_grid_search:
            # modello base: PolynomialFeatures + Ridge
            base_model = make_pipeline(PolynomialFeatures(include_bias=False), Ridge())

            # GridSearch su degree e alpha (o quello che l'utente specifica)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=kwargs.get("cv", 3),
                scoring=kwargs.get("scoring", "neg_mean_squared_error"),
                n_jobs=kwargs.get("n_jobs", None),
                verbose=kwargs.get("verbose", 0),
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            print("Best params:", grid_search.best_params_)
            print("Best CV score (scoring=neg_MSE):", grid_search.best_score_)
        else:
            # modello singolo con degree e alpha specifici
            model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=alpha))
            model.fit(X_train, y_train)

        # metriche su train
        y_pred_train = model.predict(X_train)
        j_train = mean_squared_error(y_train, y_pred_train)

        # metriche su validation (cv)
        y_pred_cv = model.predict(X_cv)
        j_cv = mean_squared_error(y_cv, y_pred_cv)

        return {
            "model": model,
            "J_train": j_train,
            "J_cv": j_cv,
            'train': (X_train, y_pred_train),
            'cv': (X_cv, y_pred_cv), 
        }


def main():
    # Dati finti: relazione NON lineare (parabola) così vedi l'effetto di degree
    # y = x^2 + rumore
    rng = np.random.RandomState(42)
    X = np.linspace(-3, 3, 200).reshape(-1, 1)  # 200 punti fra -3 e 3
    y = X[:, 0] ** 2 + rng.normal(scale=0.5, size=X.shape[0])

    # # Caso 1: modello lineare (degree=1), niente GridSearch
    # print("=== Modello lineare (degree=1, alpha=1.0) ===")
    # result_linear = CheckDataset.check_model(
    #     X=X,
    #     y=y,
    #     degree=1,
    #     alpha=1.0
    # )
    # print("J_train (lineare):", result_linear["J_train"])
    # print("J_cv    (lineare):", result_linear["J_cv"])
    # print()
    #
    # # Caso 2: modello polinomiale (degree=2), niente GridSearch
    # print("=== Modello polinomiale (degree=2, alpha=1.0) ===")
    # result_poly = CheckDataset.check_model(
    #     X=X,
    #     y=y,
    #     degree=2,
    #     alpha=1.0
    # )
    # print("J_train (poly d=2):", result_poly["J_train"])
    # print("J_cv    (poly d=2):", result_poly["J_cv"])
    # print()

    # Definisco una griglia di ricerca per degree e alpha
    param_grid = {
        "polynomialfeatures__degree": [1, 2, 3, 4],
        "ridge__alpha": [0.01, 0.1, 1.0, 10.0],
    }

    print("=== GridSearch su degree e alpha ===")
    result_grid = CheckDataset.check_model(
        X=X,
        y=y,
        param_grid=param_grid,  # attiva il ramo GridSearchCV
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        test_size=0.2,
        random_state=42,
    )

    print("\nRisultati finali sullo split esterno:")
    print("J_train:", result_grid["J_train"])
    print("J_cv   :", result_grid["J_cv"])

    # ==============================
    # Plot con pyplot per visualizzare
    # ==============================
    # griglia ordinata di x per tracciare le curve lisce

    # Modello lineare degree=1
    # model_linear = make_pipeline(
    #     PolynomialFeatures(degree=1, include_bias=False),
    #     Ridge(alpha=1.0)
    # )
    # model_linear.fit(X, y)
    # y_linear = model_linear.predict(X)
    #
    # # Modello polinomiale degree=2
    # model_poly2 = make_pipeline(
    #     PolynomialFeatures(degree=2, include_bias=False),
    #     Ridge(alpha=1.0)
    # )
    # model_poly2.fit(X, y)
    # y_poly2 = model_poly2.predict(X)

    # Scatter dei dati reali + curve dei due modelli
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="gray", alpha=0.5, label="dati")
    plt.plot(result_grid['cv'][0], result_grid['cv'][1], color="red", label="Pred CV Ridge")
    plt.plot(result_grid['train'][0], result_grid['train'][1], color="blue", label="Pred Train Ridge")
    plt.title("Confronto modello lineare vs polinomiale")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
