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
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
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
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
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
import pandas as pd

train_data = pd.read_csv("files/input/train_data.csv.zip")
test_data = pd.read_csv("files/input/test_data.csv.zip")

train_data.rename(columns={"default payment next month": "default"}, inplace=True)
test_data.rename(columns={"default payment next month": "default"}, inplace=True)

train_data.drop(columns=["ID"], inplace=True)
test_data.drop(columns=["ID"], inplace=True)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

train_data.loc[train_data["EDUCATION"] > 4, "EDUCATION"] = 4
test_data.loc[test_data["EDUCATION"] > 4, "EDUCATION"] = 4

train_data = train_data.query("MARRIAGE > 0 and EDUCATION > 0")
test_data = test_data.query("MARRIAGE > 0 and EDUCATION > 0")


# Paso 2
X_train = train_data.drop(columns=["default"])
y_train = train_data["default"]
X_test = test_data.drop(columns=["default"])
y_test = test_data["default"]


# Paso 3
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(dtype=int), categorical_features)],
    remainder="passthrough",
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)
print("Precisión:", pipeline.score(X_test, y_test))


# Paso 4
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__n_estimators": [200, 250],
    "classifier__max_depth": [10, None],
    "classifier__min_samples_split": [10],
    "classifier__min_samples_leaf": [4],
    "classifier__max_features": ["auto", "sqrt"],
    "classifier__bootstrap": [True, False],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
    verbose=True,
)

grid_search.fit(X_train, y_train)


# Paso 5
import os
import pickle
import gzip

os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:  # wb: write binary
    pickle.dump(grid_search, f)


# Paso 6
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
import json

y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)

metrics = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    },
]

output_file = "files/output/metrics.json"
os.makedirs("files/output", exist_ok=True)

with open(output_file, "w") as f:
    for item in metrics:
        f.write(str(item).replace("'", '"') + "\n")

# Paso 7
from sklearn.metrics import confusion_matrix

train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

confusion_matrices = [
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(train_cm[0, 0]),
            "predicted_1": int(train_cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(train_cm[1, 0]),
            "predicted_1": int(train_cm[1, 1]),
        },
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(test_cm[0, 0]),
            "predicted_1": int(test_cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(test_cm[1, 0]),
            "predicted_1": int(test_cm[1, 1]),
        },
    },
]

with open(output_file, "a") as f:
    for item in confusion_matrices:
        f.write(str(item).replace("'", '"') + "\n")



