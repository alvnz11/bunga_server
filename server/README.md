# 🌸 Flower Classification Server

FastAPI server untuk klasifikasi gambar bunga menggunakan Machine Learning (SVM, KNN, Manhattan Distance). Server ini terintegrasi dengan mobile app Flutter.

## 📋 Fitur

- ✅ Klasifikasi gambar bunga dengan 3 metode (SVM, KNN, Manhattan Distance)
- ✅ RESTful API dengan FastAPI
- ✅ Database SQLite untuk menyimpan history prediksi
- ✅ Ekstraksi fitur lengkap (Color, Shape, Texture, Edge)
- ✅ CORS enabled untuk integrasi dengan Flutter
- ✅ Automatic image preprocessing

## 🎯 Kelas Bunga yang Didukung

- Bellflower (风铃草)
- Rose (玫瑰)
- Sunflower (向日葵)
- Tulip (郁金香)

## 🏗️ Struktur Project

```
server/
├── main.py                  # FastAPI server
├── model_loader.py          # Load ML models
├── feature_extraction.py    # Ekstraksi fitur gambar
├── database.py             # Database handler (SQLite)
├── requirements.txt        # Python dependencies
├── models/                 # Folder untuk model files
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── reference_features.json
├── uploads/                # Folder untuk uploaded images
└── database/               # Folder untuk SQLite database
```

## 🚀 Setup & Installation

### 1. Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Export Model dari Notebook

Sebelum menjalankan server, Anda perlu export model dari notebook. Tambahkan cell berikut di akhir notebook:

```python
import joblib
import json
import numpy as np

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Save SVM model
joblib.dump(svm_best, 'models/svm_model.pkl')
print("✓ SVM model saved")

# Save KNN model
joblib.dump(knn_best, 'models/knn_model.pkl')
print("✓ KNN model saved")

# Save Scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Scaler saved")

# Save Label Encoder
joblib.dump(le, 'models/label_encoder.pkl')
print("✓ Label Encoder saved")

# Save reference features untuk Manhattan distance
# (gunakan mean features dari setiap kelas di training set)
reference_features = {}
for class_name in classes:
    class_indices = np.where(y_train == class_name)[0]
    class_features = X_train_weighted[class_indices]
    mean_features = np.mean(class_features, axis=0)
    reference_features[class_name] = mean_features.tolist()

with open('models/reference_features.json', 'w') as f:
    json.dump(reference_features, f)
print("✓ Reference features saved")

print("\n✅ All models exported successfully!")
print("📁 Copy the 'models/' folder to 'server/' directory")
```

### 3. Jalankan Server

```bash
# Development mode (auto-reload)
python main.py

# Atau menggunakan uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server akan berjalan di: `http://localhost:8000`

## 📡 API Endpoints

### 1. Health Check
```http
GET /
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

### 2. Predict Image
```http
POST /predict
Content-Type: multipart/form-data

Body:
- image: file (JPEG/PNG)
```

Response:
```json
{
  "success": true,
  "id": 1,
  "filename": "20240101_120000_flower.jpg",
  "predictions": {
    "svm": {
      "class": "rose",
      "confidence": 95.5
    },
    "knn": {
      "class": "rose",
      "confidence": 92.3
    }
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 3. Get History
```http
GET /history?limit=50
```

Response:
```json
{
  "success": true,
  "count": 10,
  "data": [
    {
      "id": 1,
      "image_filename": "20240101_120000_flower.jpg",
      "prediction_svm": "rose",
      "confidence_svm": 95.5,
      "prediction_knn": "rose",
      "confidence_knn": 92.3,
      "created_at": "2024-01-01T12:00:00"
    }
  ]
}
```

### 4. Get Prediction by ID
```http
GET /history/{prediction_id}
```

### 5. Get Statistics
```http
GET /statistics
```

Response:
```json
{
  "success": true,
  "data": {
    "total_predictions": 100,
    "class_distribution": {
      "rose": 30,
      "sunflower": 25,
      "tulip": 25,
      "bellflower": 20
    },
    "average_confidence_svm": 92.5,
    "average_confidence_knn": 89.3
  }
}
```

### 6. Delete Prediction
```http
DELETE /history/{prediction_id}
```

### 7. Get Classes
```http
GET /classes
```

Response:
```json
{
  "success": true,
  "classes": ["bellflower", "rose", "sunflower", "tulip"]
}
```

### 8. Get Model Accuracy
```http
GET /accuracy
```

Response:
```json
{
  "success": true,
  "data": {
    "svm": {
      "train_accuracy": 0.9526,
      "val_accuracy": 0.8742,
      "best_params": {
        "C": 50,
        "gamma": "scale",
        "kernel": "rbf"
      },
      "best_cv_score": 0.8852
    },
    "knn": {
      "train_accuracy": 1.0,
      "val_accuracy": 0.7877,
      "best_params": {
        "metric": "manhattan",
        "n_neighbors": 1,
        "weights": "distance"
      },
      "best_cv_score": 0.8296
    }
  }
}
```

## 📱 Integrasi dengan Flutter

### Contoh Request dari Flutter (Dart):

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> predictFlower(File imageFile) async {
  var uri = Uri.parse('http://localhost:8000/predict');
  var request = http.MultipartRequest('POST', uri);
  
  request.files.add(
    await http.MultipartFile.fromPath('image', imageFile.path)
  );
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}

// Usage:
var result = await predictFlower(imageFile);
print('SVM Prediction: ${result['predictions']['svm']['class']}');
print('Confidence: ${result['predictions']['svm']['confidence']}%');
```

## 🧪 Testing

### Test dengan cURL:

```bash
# Health check
curl http://localhost:8000/health

# Predict image
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/flower.jpg"

# Get history
curl http://localhost:8000/history?limit=10
```

### Test dengan Python:

```python
import requests

# Predict
with open('flower.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'image': f}
    )
    print(response.json())

# Get history
response = requests.get('http://localhost:8000/history?limit=10')
print(response.json())
```

## 🔧 Troubleshooting

### Models not loaded
- Pastikan folder `models/` ada dan berisi semua file yang diperlukan
- Jalankan cell export model di notebook terlebih dahulu

### Database error
- Pastikan folder `database/` ada dan writable
- Cek permission folder

### Image upload error
- Pastikan folder `uploads/` ada dan writable
- Cek file size limit (default: 16MB di FastAPI)

### CORS error dari Flutter
- CORS sudah di-enable di server dengan `allow_origins=["*"]`
- Untuk production, ganti dengan domain spesifik

## 📊 Performance

- **Preprocessing time**: ~0.5-1s per image
- **Feature extraction**: ~0.2-0.5s
- **Prediction**: ~0.1s (all 3 methods)
- **Total**: ~1-2s per request

## 🔐 Security Notes (untuk Production)

1. **CORS**: Ganti `allow_origins=["*"]` dengan domain Flutter app
2. **File Upload**: Tambahkan file size limit dan validation
3. **Rate Limiting**: Implementasi rate limiting untuk API
4. **Authentication**: Tambahkan JWT atau API key
5. **HTTPS**: Deploy dengan HTTPS (gunakan nginx + certbot)

## 📝 License

MIT License

## 👥 Author

Project Klasifikasi Bunga - Machine Learning
