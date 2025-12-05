import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.service_ia.training.under_over.service.filter_uo_service import FilterUOService

service = FilterUOService(list_stats=['mean_statistics','shots'])
dataset = service.get_match_total_goals()
df = service.split_dataset_total_goals(dataset=dataset)
X = df.drop(columns=['id_fixture', 'y', 'name_home', 'name_away']).values
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modello
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# Addestramento
model.fit(X_train, y_train)

# Predizione
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  (Errore medio assoluto): {mae:.3f}")
print(f"RMSE (Radice errore quadratico medio): {rmse:.3f}")
print(f"R²   (Coefficiente di determinazione): {r2:.3f}")


y_pred_int = np.round(y_pred)   # Arrotonda al numero di goal più vicino
acc = accuracy_score(y_test, y_pred_int)
print(f"Accuracy arrotondata: {acc:.2f}")