"""
Feature Extraction Module untuk Klasifikasi Bunga
Mengekstrak 4 jenis fitur: Color, Shape, Texture, dan Edge
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocessing gambar dengan:
    - Resize ke ukuran standar
    - CLAHE untuk contrast enhancement
    - Bilateral filter untuk noise reduction
    - Color normalization
    """
    # Resize
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    # 2. Bilateral filter
    img_denoised = cv2.bilateralFilter(img_enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. Color normalization
    img_normalized = np.zeros_like(img_denoised, dtype=np.float32)
    for i in range(3):
        channel = img_denoised[:,:,i].astype(np.float32)
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max - channel_min > 0:
            img_normalized[:,:,i] = 255 * (channel - channel_min) / (channel_max - channel_min)
        else:
            img_normalized[:,:,i] = channel
    
    img_final = img_normalized.astype(np.uint8)
    return img_final


def extract_hsv_features(img):
    """
    Ekstraksi fitur warna dari HSV, LAB, dan RGB
    Returns: 18 fitur warna
    """
    features = []
    
    # Convert ke HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        channel = img_hsv[:,:,i]
        features.extend([
            float(np.mean(channel)),
            float(np.std(channel))
        ])
    
    # Convert ke LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for i in range(3):
        channel = img_lab[:,:,i]
        features.extend([
            float(np.mean(channel)),
            float(np.std(channel))
        ])
    
    # RGB
    for i in range(3):
        channel = img[:,:,i]
        features.extend([
            float(np.mean(channel)),
            float(np.std(channel))
        ])
    
    return features


def extract_hu_moments(img):
    """
    Ekstraksi fitur bentuk menggunakan Hu Moments
    Returns: 7 fitur bentuk
    """
    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate moments
    moments = cv2.moments(binary)
    
    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform untuk stabilitas numerik
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments_log


def extract_texture_features(img):
    """
    Ekstraksi fitur tekstur menggunakan Local Binary Pattern (LBP)
    Returns: 10 fitur tekstur
    """
    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # LBP parameters
    radius = 3
    n_points = 8 * radius
    
    # Calculate LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Return top 10 bins sebagai fitur
    return hist[:10].tolist()


def extract_edge_features(img):
    """
    Ekstraksi fitur edge menggunakan Canny dan Sobel
    Returns: 6 fitur edge
    """
    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. Canny Edge Detection
    edges_canny = cv2.Canny(gray, 100, 200)
    
    # 2. Sobel (horizontal dan vertical)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate features
    features = [
        float(np.mean(edges_canny)),
        float(np.std(edges_canny)),
        float(np.mean(np.abs(sobelx))),
        float(np.std(sobelx)),
        float(np.mean(np.abs(sobely))),
        float(np.std(sobely))
    ]
    
    return features


def apply_feature_weighting(features, weights):
    """
    Menerapkan bobot pada fitur
    weights = {'color': 1.5, 'shape': 1.2, 'texture': 1.0, 'edge': 0.8}
    """
    features = np.array(features).copy()
    
    # Feature indices
    color_end = 18
    shape_end = color_end + 7
    texture_end = shape_end + 10
    edge_end = texture_end + 6
    
    # Apply weights
    features[:color_end] *= weights['color']
    features[color_end:shape_end] *= weights['shape']
    features[shape_end:texture_end] *= weights['texture']
    features[texture_end:edge_end] *= weights['edge']
    
    return features


def extract_all_features(img):
    """
    Ekstraksi semua fitur dari gambar
    Returns: Array 41 fitur (18 color + 7 shape + 10 texture + 6 edge)
    """
    # Preprocessing
    img_preprocessed = preprocess_image(img)
    
    # Extract features
    color_features = extract_hsv_features(img_preprocessed)
    shape_features = extract_hu_moments(img_preprocessed)
    texture_features = extract_texture_features(img_preprocessed)
    edge_features = extract_edge_features(img_preprocessed)
    
    # Combine
    all_features = (color_features + 
                   list(shape_features) + 
                   texture_features + 
                   edge_features)
    
    # Apply feature weighting
    feature_weights = {
        'color': 1.5,
        'shape': 1.2,
        'texture': 1.0,
        'edge': 0.8
    }
    
    weighted_features = apply_feature_weighting(all_features, feature_weights)
    
    return weighted_features
