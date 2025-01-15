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

import pandas as pd
import zipfile
import os

# Definir la ruta a los archivos zip
input_path = 'files/input/'

# Ruta de los archivos zip
train_zip = os.path.join(input_path, 'train_data.csv.zip')
test_zip = os.path.join(input_path, 'test_data.csv.zip')

# Función para cargar y limpiar los datasets
def load_and_clean_data(zip_path, output_path):
    # Extraemos el archivo zip y verificamos el contenido
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()  # Lista de archivos dentro del ZIP
        print(f"Archivos en {zip_path}: {file_list}")
        # Asumimos que el primer archivo es el que necesitamos
        csv_filename = file_list[0]
        z.extract(csv_filename, output_path)  # Extraemos el archivo
    
    # Ruta completa del archivo extraído
    extracted_file_path = os.path.join(output_path, csv_filename)
    
    # Cargar el archivo CSV
    df = pd.read_csv(extracted_file_path)
    
    # Renombrar la columna "default payment next month" a "default"
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'default'}, inplace=True)
    
    # Eliminar la columna "ID" si existe
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    
    # Eliminar registros con información no disponible
    df.dropna(inplace=True)
    
    # Reemplazar valores mayores a 4 en la columna EDUCATION por 4
    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    # Sobrescribir el archivo original con los datos limpios
    df.to_csv(extracted_file_path, index=False)
    
    return df

# Crear el directorio de salida si no existe
os.makedirs(input_path, exist_ok=True)

# Cargar y limpiar los datasets de entrenamiento y prueba
train_data_cleaned = load_and_clean_data(train_zip, input_path)
test_data_cleaned = load_and_clean_data(test_zip, input_path)

# Ver las primeras filas para asegurarnos de que los cambios se realizaron correctamente
print(train_data_cleaned.head())
print(test_data_cleaned.head())





#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

from sklearn.model_selection import train_test_split

# Función para dividir los datasets en X (características) e y (objetivo)
def split_features_and_target(df, target_column='default'):
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")
    
    # Separar características y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

# Dividir los datos de entrenamiento
X_train, y_train = split_features_and_target(train_data_cleaned)

# Dividir los datos de prueba
X_test, y_test = split_features_and_target(test_data_cleaned)

# Verificar las dimensiones de los conjuntos resultantes
print(f"Dimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")





#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Función para crear el pipeline
def create_pipeline(categorical_features):
    """
    Crea un pipeline para preprocesar datos y entrenar un modelo Random Forest.
    
    Args:
    - categorical_features (list): Lista de nombres de columnas categóricas.
    
    Returns:
    - pipeline (Pipeline): Pipeline completo.
    """
    # Transformador para variables categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputar valores faltantes
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
    ])
    
    # Transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)  # Aplicar a categóricas
        ],
        remainder='passthrough'  # Mantener las demás columnas sin cambios
    )
    
    # Pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocesamiento
        ('classifier', RandomForestClassifier(random_state=42))  # Modelo
    ])
    
    return pipeline

# Identificar las columnas categóricas en los datos de entrenamiento
categorical_features = ['EDUCATION']  # Asegúrate de añadir otras categóricas si las hay

# Crear el pipeline
pipeline = create_pipeline(categorical_features)

# Ajustar el pipeline a los datos de entrenamiento
pipeline.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
score = pipeline.score(X_test, y_test)

print(f"Puntaje del modelo en los datos de prueba: {score:.4f}")






#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Número de árboles
    'classifier__max_depth': [None, 10, 20, 30],  # Profundidad máxima de los árboles
    'classifier__min_samples_split': [2, 5, 10],  # Mínimas muestras para dividir un nodo
    'classifier__min_samples_leaf': [1, 2, 4]  # Mínimas muestras en una hoja
}

# Crear un scorer basado en la precisión balanceada
scorer = make_scorer(balanced_accuracy_score)

# Configurar GridSearchCV
grid_search = GridSearchCV(
    pipeline,  # Pipeline que queremos optimizar
    param_grid,  # Espacio de búsqueda
    cv=10,  # 10 divisiones para la validación cruzada
    scoring=scorer,  # Usar precisión balanceada como métrica
    n_jobs=-1,  # Usar todos los núcleos disponibles
    verbose=2  # Mostrar información sobre el progreso
)

# Ajustar GridSearchCV a los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros y el mejor puntaje
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Mejores hiperparámetros:", best_params)
print(f"Mejor precisión balanceada durante la validación cruzada: {best_score:.4f}")

# Evaluar el modelo óptimo en el conjunto de prueba
test_score = balanced_accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
print(f"Precisión balanceada en los datos de prueba: {test_score:.4f}")





#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

import gzip
import pickle
import os

# Obtener el mejor modelo del GridSearchCV
best_model = grid_search.best_estimator_

# Definir el nombre del archivo y la ruta
model_filename = 'files/models/model.pkl.gz'

# Crear el directorio si no existe
os.makedirs(os.path.dirname(model_filename), exist_ok=True)

# Guardar el modelo comprimido en un archivo .pkl.gz
with gzip.open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"Modelo guardado correctamente en {model_filename}")





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

import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

# Crear la carpeta de salida si no existe
output_dir = "files/output/"
os.makedirs(output_dir, exist_ok=True)

# Predecir en los conjuntos de entrenamiento y prueba
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calcular las métricas para el conjunto de entrenamiento
train_metrics = {
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred, zero_division=0),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred, zero_division=0),
    'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
}

# Calcular las métricas para el conjunto de prueba
test_metrics = {
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred, zero_division=0),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
}

# Guardar las métricas en un archivo JSON
metrics_filename = os.path.join(output_dir, "metrics.json")
with open(metrics_filename, 'w') as f:
    json.dump([train_metrics, test_metrics], f, indent=4)

print(f"Métricas guardadas correctamente en {metrics_filename}")





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
import json
from sklearn.metrics import confusion_matrix

# Crear una función para convertir la matriz de confusión a un formato dict
def confusion_matrix_to_dict(cm, dataset_name):
    cm_dict = {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {
            "predicted_0": int(cm[0, 0]), 
            "predicted_1": int(cm[0, 1])
        },
        'true_1': {
            "predicted_0": int(cm[1, 0]), 
            "predicted_1": int(cm[1, 1])
        }
    }
    return cm_dict

# Calcular las matrices de confusión para los conjuntos de entrenamiento y prueba
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Convertir las matrices de confusión a formato dict
train_cm_dict = confusion_matrix_to_dict(train_cm, 'train')
test_cm_dict = confusion_matrix_to_dict(test_cm, 'test')

# Leer las métricas existentes del archivo (si existe)
output_file = 'files/output/metrics.json'
try:
    with open(output_file, 'r') as f:
        metrics_data = json.load(f)
except FileNotFoundError:
    metrics_data = []

# Agregar las matrices de confusión a los datos
metrics_data.append(train_cm_dict)
metrics_data.append(test_cm_dict)

# Guardar las métricas actualizadas en el archivo JSON
with open(output_file, 'w') as f:
    json.dump(metrics_data, f, indent=4)

print(f"Matrices de confusión guardadas correctamente en {output_file}")


