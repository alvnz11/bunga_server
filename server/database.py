"""
Database Module untuk menyimpan history klasifikasi
Menggunakan SQLite
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Base class untuk models
Base = declarative_base()


class PredictionHistory(Base):
    """Model untuk menyimpan history prediksi"""
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    
    # SVM Prediction
    prediction_svm = Column(String(100), nullable=False)
    confidence_svm = Column(Float, nullable=False)
    
    # KNN Prediction
    prediction_knn = Column(String(100), nullable=False)
    confidence_knn = Column(Float, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'image_filename': self.image_filename,
            'image_path': self.image_path,
            'prediction_svm': self.prediction_svm,
            'confidence_svm': round(self.confidence_svm, 2),
            'prediction_knn': self.prediction_knn,
            'confidence_knn': round(self.confidence_knn, 2),
            'created_at': self.created_at.isoformat()
        }


class Database:
    """Database handler class"""
    
    def __init__(self, db_path='database/flower_classification.db'):
        """
        Initialize database
        Args:
            db_path: Path ke database file
        """
        # Create directory if not exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create engine
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        print(f"✓ Database initialized at {db_path}")
    
    def save_prediction(self, image_filename, image_path, 
                       prediction_svm, confidence_svm,
                       prediction_knn, confidence_knn):
        """
        Save prediction result to database
        """
        try:
            prediction = PredictionHistory(
                image_filename=image_filename,
                image_path=image_path,
                prediction_svm=prediction_svm,
                confidence_svm=confidence_svm,
                prediction_knn=prediction_knn,
                confidence_knn=confidence_knn
            )
            
            self.session.add(prediction)
            self.session.commit()
            
            return prediction.to_dict()
            
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error saving prediction: {e}")
    
    def get_all_predictions(self, limit=100):
        """
        Get all prediction history
        Args:
            limit: Maximum number of records to return
        """
        try:
            predictions = self.session.query(PredictionHistory)\
                .order_by(PredictionHistory.created_at.desc())\
                .limit(limit)\
                .all()
            
            return [pred.to_dict() for pred in predictions]
            
        except Exception as e:
            raise Exception(f"Error getting predictions: {e}")
    
    def get_prediction_by_id(self, prediction_id):
        """Get single prediction by ID"""
        try:
            prediction = self.session.query(PredictionHistory)\
                .filter(PredictionHistory.id == prediction_id)\
                .first()
            
            if prediction:
                return prediction.to_dict()
            return None
            
        except Exception as e:
            raise Exception(f"Error getting prediction: {e}")
    
    def get_statistics(self):
        """Get statistics dari prediction history"""
        try:
            from sqlalchemy import func
            
            # Total predictions
            total = self.session.query(func.count(PredictionHistory.id)).scalar()
            
            # Class distribution (SVM)
            class_dist = self.session.query(
                PredictionHistory.prediction_svm,
                func.count(PredictionHistory.prediction_svm)
            ).group_by(PredictionHistory.prediction_svm).all()
            
            # Average confidence
            avg_confidence_svm = self.session.query(
                func.avg(PredictionHistory.confidence_svm)
            ).scalar()
            
            avg_confidence_knn = self.session.query(
                func.avg(PredictionHistory.confidence_knn)
            ).scalar()
            
            return {
                'total_predictions': total,
                'class_distribution': {cls: count for cls, count in class_dist},
                'average_confidence_svm': round(avg_confidence_svm or 0, 2),
                'average_confidence_knn': round(avg_confidence_knn or 0, 2)
            }
            
        except Exception as e:
            raise Exception(f"Error getting statistics: {e}")
    
    def delete_prediction(self, prediction_id):
        """Delete prediction by ID"""
        try:
            prediction = self.session.query(PredictionHistory)\
                .filter(PredictionHistory.id == prediction_id)\
                .first()
            
            if prediction:
                self.session.delete(prediction)
                self.session.commit()
                return True
            return False
            
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error deleting prediction: {e}")
    
    def close(self):
        """Close database connection"""
        self.session.close()
