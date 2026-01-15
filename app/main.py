from flask import Flask, request, jsonify
import mlflow.pyfunc
import random
import pandas as pd
import datetime
import os

app = Flask(__name__)

# Параметры автоматизации и связи с реестром моделей
CONFIG = {
    "traffic_ratio_b": 0.3,  
    "model_name": "iris_model",
    "alias_a": "champion", 
    "alias_b": "challenger"
}

# Настройка подключения к центральному серверу Mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Автоматическая загрузка моделей из Mlflow
def get_model(alias):
    try:
        model_uri = f"models:/{CONFIG['model_name']}@{alias}"
        print(f"Attempting to load model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"CRITICAL ERROR loading alias {alias}: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json.get('data') 
    if not input_data:
        return jsonify({"error": "No data provided"}), 400
        
    # Корректировка имен признаков
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    df = pd.DataFrame([input_data], columns=columns)
    
    # A/B логика
    if random.random() < CONFIG["traffic_ratio_b"]:
        current_alias = CONFIG["alias_b"]
        model_variant = 'B'
    else:
        current_alias = CONFIG["alias_a"]
        model_variant = 'A'

    model = get_model(current_alias)
    
    # Запасная модель
    if model is None:
        fallback_alias = CONFIG["alias_a"] if current_alias == CONFIG["alias_b"] else CONFIG["alias_b"]
        print(f"Primary model {current_alias} failed. Trying fallback {fallback_alias}...")
        model = get_model(fallback_alias)
        model_variant = f"Fallback ({fallback_alias})"

    if model is None:
        return jsonify({
            "error": "All models failed to load.",
            "attempted_aliases": [CONFIG["alias_a"], CONFIG["alias_b"]]
        }), 500

    prediction = model.predict(df)[0]
    
    # Сбор данных для анализа
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "variant": model_variant,
        "input": str(input_data),
        "prediction": int(prediction)
    }
    
    LOG_DIR = '/app/data'
    LOG_FILE = os.path.join(LOG_DIR, 'ab_logs.csv')
    os.makedirs(LOG_DIR, exist_ok=True)
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
    
    return jsonify({
        "variant": model_variant, 
        "prediction": int(prediction),
        "model_used": current_alias if "Fallback" not in model_variant else "fallback"
    })


# Возможность динамической смены пропорций трафика
@app.route('/set_traffic', methods=['POST'])
def set_traffic():
    ratio = request.json.get('ratio')
    if ratio is not None and 0 <= ratio <= 1:
        CONFIG["traffic_ratio_b"] = ratio
        return jsonify({"status": "success", "new_ratio_b": ratio})
    return jsonify({"status": "error", "message": "Invalid ratio. Use 0 to 1"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)