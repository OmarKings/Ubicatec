import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("DEBUG: Ejecutando convert_to_onnx.py")

MODEL_PATH = r"C:\Users\OmarKings\Desktop\lidar\model.joblib"
OUTPUT_PATH = r"C:\Users\OmarKings\Desktop\lidar\model.onnx"

print("Cargando modelo desde:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("ERROR: No existe model.joblib")
    exit()

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]

print("Convirtiendo el modelo a ONNX...")

initial_type = [('input', FloatTensorType([None, 42]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

print("Guardando archivo ONNX en:", OUTPUT_PATH)

with open(OUTPUT_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("CONVERSION COMPLETADA EXITOSAMENTE")
