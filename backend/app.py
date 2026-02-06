import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Thêm đường dẫn đến thư mục gốc của project (để import được methods/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# ===================== ROUTES =====================

@app.route('/')
def home():
    return "Deepfake Detection API is running!"


@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({'message': 'API is working!', 'status': 'success'})


@app.route('/api/methods', methods=['GET'])
def get_methods():
    """Lấy danh sách các phương pháp available"""
    methods = [
        {
            'code': 'A',
            'name': 'Frame-wise Baseline',
            'description': 'Phân tích từng khung hình độc lập'
        },
        {
            'code': 'B',
            'name': 'Frame CNN + Voting',
            'description': 'Kết hợp CNN với voting theo thời gian'
        },
        {
            'code': 'C',
            'name': 'Feature Aggregation',
            'description': 'Trích xuất đặc trưng tổng hợp + MLP'
        },
        {
            'code': 'D',
            'name': 'Patch-level + Temporal',
            'description': 'Patch-level feature + Temporal model (LSTM/Transformer)'
        }
    ]
    return jsonify(methods)


@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    try:
        print("Received detection request")

        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        method = request.form.get('method', 'A')

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Lưu file tạm
        filename = file.filename
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        print(f"File saved: {filename}, Method: {method}")

        # ================== GỌI MODEL THẬT ==================
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # if method == 'A':
        #     from methods.A import FrameWiseDetector
        #     model_path = os.path.join(base_dir, "outputs_frameA_resnetrs50", "frame_baseline_best.pt")
        #     detector = FrameWiseDetector(model_path)
        #     result = detector.predict_video(temp_path)
        #
        # elif method == 'B':
        #     from methods.B import SlidingWindowDetector
        #     model_path = os.path.join(base_dir, "outputs_frameB_resnetrs50", "frame_baseline_best.pt")
        #     detector = SlidingWindowDetector(model_path)
        #     result = detector.predict_video(temp_path)
        #
        # elif method == 'C':
        #     from methods.C import FeatureAggregationDetector
        #     model_path = os.path.join(base_dir, "outputs_frameC_resnetrs50", "feature_agg_max_mean_best.pt")
        #     detector = FeatureAggregationDetector(model_path)
        #     result = detector.predict_video(temp_path)

        if method == 'D':
            from methods.D import PatchTemporalDetector
            model_path = os.path.join(base_dir, "outputs_frameD_resnetrs50", "proposed_model_best.pth")
            detector = PatchTemporalDetector(model_path)
            result = detector.predict_video(temp_path)
        else:
            return jsonify({'error': 'Invalid method'}), 400

        # Xóa file tạm sau khi xử lý
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'is_fake': bool(result['is_fake']),
            'confidence': float(result['confidence']),
            'method_used': f'Method {method}',
            'method_code': method
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

        # Đảm bảo xóa file nếu có lỗi
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500


# ===================== RUN APP =====================

if __name__ == '__main__':
    print("Starting Deepfake Detection API...")
    print("Available routes:")
    print("- GET  /api/test")
    print("- GET  /api/methods")
    print("- POST /api/detect")
    app.run(debug=True, port=5000, host='0.0.0.0')
