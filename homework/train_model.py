import pandas as pd
import zipfile
import os
import json
import gzip
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer

# Ruta de los archivos ZIP
input_path = 'files/input/'
train_zip = os.path.join(input_path, 'train_data.csv.zip')
test_zip = os.path.join(input_path, 'test_data.csv.zip')

# Paso 1: Función para cargar y limpiar los datos
def load_and_clean_data(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        print(f"Archivos en {zip_path}: {file_list}")
        csv_filename = file_list[0]
        z.extract(csv_filename, output_path)
    
    df = pd.read_csv(os.path.join(output_path, csv_filename))
    
    # Limpiar los datos
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns=['ID'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    df.to_csv(os.path.join(output_path, csv_filename), index=False)
    return df

# Cargar y limpiar los datos de entrenamiento y prueba
train_data_cleaned = load_and_clean_data(train_zip, input_path)
test_data_cleaned = load_and_clean_data(test_zip, input_path)

# Paso 2: Dividir los datos en características (X) y objetivo (y)
def split_features_and_target(df, target_column='default'):
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

X_train, y_train = split_features_and_target(train_data_cleaned)
X_test, y_test = split_features_and_target(test_data_cleaned)

# Paso 3: Crear el pipeline
def create_pipeline(categorical_features):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline

# Identificar las columnas categóricas
categorical_features = ['EDUCATION']
pipeline = create_pipeline(categorical_features)

# Ajustar el modelo
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Puntaje del modelo en los datos de prueba: {score:.4f}")

# Paso 4: Optimizar los hiperparámetros
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
scorer = make_scorer(balanced_accuracy_score)

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring=scorer, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Mejores hiperparámetros: {best_params}")
print(f"Mejor precisión balanceada: {best_score:.4f}")

test_score = balanced_accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
print(f"Precisión balanceada en los datos de prueba: {test_score:.4f}")

# Paso 5: Guardar el modelo
model_filename = 'files/models/model.pkl.gz'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
with gzip.open(model_filename, 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)
print(f"Modelo guardado correctamente en {model_filename}")

# Paso 6: Calcular métricas y guardarlas en metrics.json
def calculate_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

train_metrics = {'dataset': 'train', **calculate_metrics(y_train, y_train_pred)}
test_metrics = {'dataset': 'test', **calculate_metrics(y_test, y_test_pred)}

metrics_filename = 'files/output/metrics.json'
os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

metrics_data = [train_metrics, test_metrics]

with open(metrics_filename, 'w') as f:
    json.dump(metrics_data, f, indent=4)
print(f"Métricas guardadas correctamente en {metrics_filename}")

# Paso 7: Calcular matrices de confusión y guardarlas
def confusion_matrix_to_dict(cm, dataset_name):
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

train_cm_dict = confusion_matrix_to_dict(train_cm, 'train')
test_cm_dict = confusion_matrix_to_dict(test_cm, 'test')

try:
    with open(metrics_filename, 'r') as f:
        metrics_data = json.load(f)
except FileNotFoundError:
    metrics_data = []

metrics_data.append(train_cm_dict)
metrics_data.append(test_cm_dict)

with open(metrics_filename, 'w') as f:
    json.dump(metrics_data, f, indent=4)
print(f"Matrices de confusión guardadas correctamente en {metrics_filename}")
