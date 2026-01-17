from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow import keras
import io
import os

app = Flask(__name__)
CORS(app)

# Đường dẫn đến model
MODEL_PATH = 'model/keras_model.h5'
LABELS_PATH = 'model/labels.txt'

# Load model
print("Đang load model...")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Lỗi khi load model: {e}")
    model = None

# Đọc labels
def load_labels():
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        return labels
    except Exception as e:
        print(f"✗ Lỗi khi đọc labels: {e}")
        # Fallback nếu không có file labels.txt
        return ['Táo', 'Chuối', 'Không có gì']

class_names = load_labels()
print(f"✓ Loaded {len(class_names)} classes: {class_names}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model chưa được load'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
    
    try:
        # Đọc ảnh
        image = Image.open(io.BytesIO(file.read()))
        
        # Chuyển sang RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize về 224x224 (kích thước chuẩn của Teachable Machine)
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Chuyển thành numpy array
        image_array = np.asarray(image, dtype=np.float32)
        
        # Chuẩn hóa theo cách Teachable Machine
        # Teachable Machine normalize từ 0-255 về -1 đến 1
        normalized_image = (image_array / 127.5) - 1
        
        # Reshape để phù hợp với input model (batch_size, height, width, channels)
        data = np.expand_dims(normalized_image, axis=0)
        
        # Dự đoán
        predictions = model.predict(data, verbose=0)[0]
        
        # Tạo kết quả
        results = []
        for i, confidence in enumerate(predictions):
            if i < len(class_names):
                results.append({
                    'class_name': class_names[i],
                    'confidence': float(confidence)
                })
        
        # Tìm prediction cao nhất
        top_prediction = max(results, key=lambda x: x['confidence'])
        
        # Thêm logic để xử lý "Không có gì"
        if top_prediction['confidence'] < 0.5:  # Ngưỡng confidence thấp
            message = "Không chắc chắn. Vui lòng chụp rõ hơn!"
        elif 'Không có gì' in top_prediction['class_name'] or 'None' in top_prediction['class_name']:
            message = "Không phát hiện trái cây trong ảnh"
        else:
            message = f"Phát hiện: {top_prediction['class_name']} ({top_prediction['confidence']*100:.1f}%)"
        
        # Log kết quả
        print(f"Prediction: {message}")
        
        return jsonify({
            'predictions': results,
            'message': message,
            'top_prediction': top_prediction,
            'success': True
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint để kiểm tra server có hoạt động không"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': class_names,
        'num_classes': len(class_names)
    })

if __name__ == '__main__':
    # Kiểm tra file có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ Warning: Model file not found at {MODEL_PATH}")
        print("   Vui lòng đặt file keras_model.h5 vào thư mục model/")
    if not os.path.exists(LABELS_PATH):
        print(f"⚠ Warning: Labels file not found at {LABELS_PATH}")
        print("   Vui lòng đặt file labels.txt vào thư mục model/")
    
    print("\n🚀 Starting Flask server...")
    print(f"📍 Server will run at: http://localhost:5000")
    print(f"📷 Open browser and go to: http://localhost:5000")
    print(f"🔍 Health check: http://localhost:5000/health")
    print("\nPress CTRL+C to quit\n")
    
    # app.run(debug=True, port=5000, host='0.0.0.0')
   
    app.run(host='0.0.0.0', port=5000)
