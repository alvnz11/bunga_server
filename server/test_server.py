"""
Script untuk testing FastAPI server
Test semua endpoint dan fungsionalitas
"""

import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_health():
    """Test health check endpoint"""
    print_header("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy!")
            print(f"   Models loaded: {data.get('models_loaded')}")
            print(f"   SVM loaded: {data.get('svm_loaded')}")
            print(f"   KNN loaded: {data.get('knn_loaded')}")
            print(f"   Scaler loaded: {data.get('scaler_loaded')}")
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Server tidak berjalan!")
        print("   Jalankan: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_root():
    """Test root endpoint"""
    print_header("TEST 2: Root Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint OK")
            print(f"   Message: {data.get('message')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_classes():
    """Test get classes endpoint"""
    print_header("TEST 3: Get Classes")
    
    try:
        response = requests.get(f"{BASE_URL}/classes", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Classes retrieved successfully")
            print(f"   Classes: {', '.join(data.get('classes', []))}")
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict(image_path=None):
    """Test predict endpoint"""
    print_header("TEST 4: Predict Image")
    
    # Jika tidak ada image path, gunakan rose.jpg
    if not image_path:
        image_path = "../data/rose.jpg"
        
        # Jika rose.jpg tidak ada, cari di folder rose
        if not os.path.exists(image_path):
            rose_folder = Path("../data/rose")
            if rose_folder.exists():
                images = list(rose_folder.glob("*.jpg")) + list(rose_folder.glob("*.png"))
                if images:
                    image_path = str(images[0])
    
    if not image_path or not os.path.exists(image_path):
        print("⚠ Tidak ada sample image untuk di-test")
        print("  Pastikan file rose.jpg ada di folder data/")
        print("  Atau jalankan dengan: python test_server.py /path/to/image.jpg")
        return False
    
    print(f"📸 Testing dengan image: {Path(image_path).name}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{BASE_URL}/predict", 
                files=files, 
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful!")
            print(f"\n   📊 Results:")
            
            # SVM
            svm = data['predictions']['svm']
            print(f"   SVM:       {svm['class']:12} (confidence: {svm['confidence']:.1f}%)")
            
            # KNN
            knn = data['predictions']['knn']
            print(f"   KNN:       {knn['class']:12} (confidence: {knn['confidence']:.1f}%)")
            
            print(f"\n   🆔 Prediction ID: {data['id']}")
            return data['id']
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_history():
    """Test history endpoint"""
    print_header("TEST 5: Get History")
    
    try:
        response = requests.get(f"{BASE_URL}/history?limit=5", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ History retrieved successfully")
            print(f"   Total records: {data['count']}")
            
            if data['count'] > 0:
                print("\n   📝 Recent predictions:")
                for pred in data['data'][:3]:
                    print(f"      ID {pred['id']}: {pred['prediction_svm']} ({pred['image_filename']})")
            
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_accuracy():
    """Test accuracy endpoint"""
    print_header("TEST 6: Get Accuracy Metrics")
    
    try:
        response = requests.get(f"{BASE_URL}/accuracy", timeout=5)
        
        if response.status_code == 200:
            data = response.json()['data']
            print("✅ Accuracy metrics retrieved successfully")
            print(f"\n   🎯 SVM Accuracy:")
            print(f"   Training:   {data['svm']['train_accuracy']:.2%}")
            print(f"   Validation: {data['svm']['val_accuracy']:.2%}")
            print(f"   CV Score:   {data['svm']['best_cv_score']:.2%}")
            
            print(f"\n   🎯 KNN Accuracy:")
            print(f"   Training:   {data['knn']['train_accuracy']:.2%}")
            print(f"   Validation: {data['knn']['val_accuracy']:.2%}")
            print(f"   CV Score:   {data['knn']['best_cv_score']:.2%}")
            
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_statistics():
    """Test statistics endpoint"""
    print_header("TEST 7: Get Statistics")
    
    try:
        response = requests.get(f"{BASE_URL}/statistics", timeout=5)
        
        if response.status_code == 200:
            data = response.json()['data']
            print("✅ Statistics retrieved successfully")
            print(f"\n   📊 Statistics:")
            print(f"   Total predictions: {data['total_predictions']}")
            print(f"   Avg confidence (SVM): {data['average_confidence_svm']:.2f}%")
            print(f"   Avg confidence (KNN): {data['average_confidence_knn']:.2f}%")
            
            if data['class_distribution']:
                print(f"\n   📈 Class Distribution:")
                for cls, count in data['class_distribution'].items():
                    print(f"      {cls:12} → {count} predictions")
            
            return True
        else:
            print(f"❌ Error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests(image_path=None):
    """Run all tests"""
    print("\n" + "🧪 "*35)
    print("  FLOWER CLASSIFICATION SERVER - TEST SUITE")
    print("🧪 "*35)
    
    results = {
        'health': test_health(),
        'root': test_root(),
        'classes': test_classes(),
        'predict': test_predict(image_path) is not None,
        'history': test_history(),
        'accuracy': test_accuracy(),
        'statistics': test_statistics()
    }
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:15} {status}")
    
    print("\n" + "="*70)
    print(f"  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("  🎉 All tests passed! Server is ready for production.")
    else:
        print("  ⚠ Some tests failed. Please check the logs above.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Get image path from command line if provided
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run all tests
    run_all_tests(image_path)
