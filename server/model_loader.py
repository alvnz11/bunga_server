"""
Model Loader Module
Load model SVM, KNN, Scaler, dan Label Encoder
"""

import joblib
import numpy as np
from pathlib import Path


class ModelLoader:
    def __init__(self, models_dir='models'):
        """
        Initialize ModelLoader
        Args:
            models_dir: Directory tempat model disimpan
        """
        self.models_dir = Path(models_dir)
        self.svm_model = None
        self.knn_model = None
        self.scaler = None
        self.label_encoder = None
        
    def load_models(self):
        """Load semua model yang diperlukan"""
        try:
            # Load SVM model
            svm_path = self.models_dir / 'svm_model.pkl'
            if svm_path.exists():
                self.svm_model = joblib.load(svm_path)
                print(f"✓ SVM model loaded from {svm_path}")
            else:
                print(f"⚠ SVM model not found at {svm_path}")
            
            # Load KNN model
            knn_path = self.models_dir / 'knn_model.pkl'
            if knn_path.exists():
                self.knn_model = joblib.load(knn_path)
                print(f"✓ KNN model loaded from {knn_path}")
            else:
                print(f"⚠ KNN model not found at {knn_path}")
            
            # Load Scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded from {scaler_path}")
            else:
                print(f"⚠ Scaler not found at {scaler_path}")
            
            # Load Label Encoder
            le_path = self.models_dir / 'label_encoder.pkl'
            if le_path.exists():
                self.label_encoder = joblib.load(le_path)
                print(f"✓ Label Encoder loaded from {le_path}")
            else:
                print(f"⚠ Label Encoder not found at {le_path}")
                
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
    
    def predict_svm(self, features):
        """
        Prediksi menggunakan SVM
        Returns: (class_name, confidence_percentage)
        """
        if self.svm_model is None or self.scaler is None or self.label_encoder is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Debug scaler
        print(f"DEBUG - Scaler mean (first 10): {self.scaler.mean_[:10]}")
        print(f"DEBUG - Scaler std (first 10): {self.scaler.scale_[:10]}")
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Debug
        print(f"DEBUG - Scaled features (first 10): {features_scaled[0][:10]}")
        print(f"DEBUG - Scaled stats - mean: {np.mean(features_scaled[0]):.6f}, std: {np.std(features_scaled[0]):.6f}")
        
        # Predict
        prediction_encoded = self.svm_model.predict(features_scaled)[0]
        class_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (probability)
        if hasattr(self.svm_model, 'predict_proba'):
            probabilities = self.svm_model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities) * 100)
            print(f"DEBUG - SVM probabilities: {probabilities}")
        else:
            # If no predict_proba, use decision_function
            decision_values = self.svm_model.decision_function(features_scaled)[0]
            print(f"DEBUG - SVM decision values: {decision_values}")
            print(f"DEBUG - SVM decision argmax: {np.argmax(decision_values)}")
            print(f"DEBUG - Label encoder classes: {self.label_encoder.classes_}")
            # Convert to pseudo-probability
            max_decision = np.max(decision_values)
            confidence = float(min(100, max(50, (max_decision + 5) * 10)))
        
        return class_name, confidence
    
    def predict_knn(self, features):
        """
        Prediksi menggunakan KNN
        Returns: (class_name, confidence_percentage)
        """
        if self.knn_model is None or self.scaler is None or self.label_encoder is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction_encoded = self.knn_model.predict(features_scaled)[0]
        class_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (probability)
        probabilities = self.knn_model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities) * 100)
        
        return class_name, confidence
    
    def get_class_names(self):
        """Get list of class names"""
        if self.label_encoder is None:
            return []
        return self.label_encoder.classes_.tolist()
