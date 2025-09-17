# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


# Paso 1

import zipfile
import os
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score

def custom_scorer(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    # Combinamos ambas métricas dando más peso a la precisión
    return 0.7 * precision + 0.3 * balanced_acc

custom_score = make_scorer(custom_scorer)
import gzip
import pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score





zip_path1 = "files/input/train_data.csv.zip"
zip_path2 = "files/input/test_data.csv.zip"
extract_path = "files/input/"

if not os.path.exists("files/input/train_default_of_credit_card_clients.csv"):
  with zipfile.ZipFile(zip_path1, 'r') as zip_ref:
      zip_ref.extractall(extract_path)

if not os.path.exists("files/input/test_default_of_credit_card_clients.csv"):
  with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

train = pd.read_csv("files/input/train_default_of_credit_card_clients.csv")
test = pd.read_csv("files/input/test_default_of_credit_card_clients.csv")

train = train.rename(columns={"default payment next month": "default"})
test = test.rename(columns={"default payment next month": "default"})

train = train.drop(columns=["ID"])
test = test.drop(columns=["ID"])


train["EDUCATION"] = train["EDUCATION"].replace(0, np.nan)
test["EDUCATION"] = test["EDUCATION"].replace(0, np.nan)

train = train.dropna()
test = test.dropna()

train.loc[train["EDUCATION"] > 4, "EDUCATION"] = 4
test.loc[test["EDUCATION"] > 4, "EDUCATION"] = 4

train = train.drop_duplicates()
test = test.drop_duplicates()

# Paso 2
x_train = train.drop("default", axis=1)
y_train = train["default"]

x_test = test.drop("default", axis=1)
y_test = test["default"]

# Paso 3
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

numeric_cols = [c for c in x_train.columns if c not in categorical_cols]

categorical_pipeline = Pipeline(steps=[
   ("imputer", SimpleImputer(strategy="most_frequent")),
   ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline(steps=[
   ("imputer",SimpleImputer(strategy="median")),
   ("scaler", MinMaxScaler())
])

preprocessor = ColumnTransformer(transformers=[
   ("cat", categorical_pipeline, categorical_cols),
   ("num", numeric_pipeline, numeric_cols)
])

clf_pipeline = Pipeline(steps=[
   ("preprocessor", preprocessor),
   ("feature_selection", SelectKBest(score_func=f_classif, k=25)),
   ("clf", LogisticRegression(random_state=42, max_iter=2000))
])

param_grid = {
    "feature_selection__k": ["all"],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"],
}

grid_search = GridSearchCV(
   estimator=clf_pipeline,
   param_grid=param_grid,
   cv=2,
   scoring='balanced_accuracy',  # Use built-in scorer for serialization
   n_jobs=-1,
   verbose=2
)

grid_search.fit(x_train, y_train)

print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor balanced accuracy (CV):", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

print("Balanced Accuracy en test:",
      balanced_accuracy_score(y_test, y_pred))

# Paso 5: Guardar el modelo comprimido con gzip
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)  # Save the entire GridSearchCV object

# Cargar el modelo guardado para verificar
with gzip.open("files/models/model.pkl.gz", "rb") as f:
    loaded_model = pickle.load(f)

print(type(loaded_model))
print(loaded_model.best_estimator_.predict(x_test[:5]))

def compute_metrics(model, X, y, dataset_name):
    y_pred = model.predict(X)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0)
    }

metrics_train = compute_metrics(best_model, x_train, y_train, "train")
metrics_test  = compute_metrics(best_model, x_test, y_test, "test")


def compute_confusion_matrix(model, X, y, dataset_name):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

cm_train = compute_confusion_matrix(best_model, x_train, y_train, "train")
cm_test  = compute_confusion_matrix(best_model, x_test, y_test, "test")

results = [metrics_train, metrics_test, cm_train, cm_test]

output_path = "files/output/metrics.json"
with open(output_path, "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

print("Métricas guardadas en:", output_path)
