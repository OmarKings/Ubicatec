import os
import numpy as np
import joblib  # Better than pickle for sklearn models
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings for clean output

DATA_DIR = './data'

# Carga y validación de datos
data = []
labels = []
class_sample_counts = Counter()  # Track samples per class

print(" Cargando datos...")
for dir_ in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(path):
        continue
    
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    if not files:
        print(f"  Advertencia: Carpeta '{dir_}' vacía. Saltando.")
        continue
    
    for file in files:
        npy_path = os.path.join(path, file)
        try:
            landmark = np.load(npy_path)
            if landmark.shape != (42,):  # Validate shape (21 landmarks * 2 coords)
                print(f"  Advertencia: {file} tiene forma {landmark.shape}, esperado (42,). Saltando.")
                continue
            data.append(landmark)
            labels.append(dir_)
            class_sample_counts[dir_] += 1
        except Exception as e:
            print(f" Error cargando {npy_path}: {e}")
            continue

if not data:
    print(" No se encontraron datos válidos. Asegúrate de haber recolectado muestras.")
    exit(1)

# Convert to arrays for efficiency
X = np.array(data)
y = np.array(labels)

# Check balance and warn
print("\n Distribución de clases:")
for cls, count in class_sample_counts.items():
    print(f"  {cls}: {count} muestras")
    if count < 50:
        print(f"      Bajo: <50 muestras para '{cls}' – considera recolectar más para mejor precisión.")

min_samples = min(class_sample_counts.values())
if min_samples < 20:
    print(f"\n  Clase con menos muestras: {min_samples}. Precisión podría ser baja (<70%).")

# Divide los datos (20% test, stratified)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42  # Added random_state for reproducibility
)

print(f"\n Datos cargados: {len(X)} totales | Train: {len(x_train)} | Test: {len(x_test)}")

# Entrena modelo con tuning básico
print("\n Entrenando modelo con búsqueda de hiperparámetros...")
param_grid = {
    'n_estimators': [100, 200],  # Your original + one more
    'max_depth': [None, 10, 20],  # Prevent overfitting
    'random_state': [42]
}
rf_base = RandomForestClassifier()
grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1)  # 3-fold CV for tuning
grid_search.fit(x_train, y_train)

model = grid_search.best_estimator_
print(f" Mejores params: {grid_search.best_params_}")

# Cross-validation para estimación robusta
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print(f" Cross-val accuracy (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Evalúa en test set
y_pred = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print(f"\n Precisión en test: {accuracy*100:.2f}%")

# Métricas detalladas
print("\n Reporte de clasificación (F1-score por clase):")
print(classification_report(y_test, y_pred))

# Matriz de confusión (opcional: guarda como imagen si usas matplotlib)
print("\n Matriz de confusión (filas=real, cols=pred):")
print(confusion_matrix(y_test, y_pred))

# Feature importance (útil para depurar: ¿qué landmarks importan?)
if hasattr(model, 'feature_importances_'):
    top_features = np.argsort(model.feature_importances_)[-5:]  # Top 5
    print(f"\n Top 5 landmarks más importantes (índices): {top_features}")

# Guarda modelo con metadata
model_data = {
    'model': model,
    'classes': sorted(np.unique(y)),  # Lista de letras para predicción
    'feature_names': [f'lm{i//2}_x' if i%2==0 else f'lm{i//2}_y' for i in range(42)]  # Opcional: nombres de features
}
try:
    joblib.dump(model_data, 'model.joblib')
    print("\n Modelo guardado como model.joblib (usa joblib.load para cargar).")
except Exception as e:
    print(f" Error al guardar: {e}")

print("\n Entrenamiento completado. Para predecir en tiempo real, usa un script con cv2 + MediaPipe.")
