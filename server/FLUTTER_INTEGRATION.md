# 📱 Flutter Integration Guide

## Setup HTTP Package

Tambahkan dependencies di `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.1.0
  image_picker: ^1.0.4
  path_provider: ^2.1.1
```

## API Service Class

Buat file `lib/services/flower_api_service.dart`:

```dart
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

class FlowerApiService {
  // Ganti dengan IP server Anda
  // Untuk emulator Android: 10.0.2.2
  // Untuk device fisik: IP komputer di network yang sama
  static const String baseUrl = 'http://10.0.2.2:8000';
  
  /// Health check
  Future<bool> isServerHealthy() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['models_loaded'] == true;
      }
      return false;
    } catch (e) {
      print('Health check error: $e');
      return false;
    }
  }
  
  /// Predict flower from image
  Future<Map<String, dynamic>> predictFlower(File imageFile) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/predict'),
      );
      
      // Add image file
      request.files.add(
        await http.MultipartFile.fromPath('image', imageFile.path),
      );
      
      // Send request
      var streamedResponse = await request.send()
          .timeout(const Duration(seconds: 30));
      
      // Get response
      var response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Prediction failed: ${response.body}');
      }
    } catch (e) {
      throw Exception('Error during prediction: $e');
    }
  }
  
  /// Get prediction history
  Future<List<dynamic>> getHistory({int limit = 50}) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/history?limit=$limit'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 10));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['data'];
      } else {
        throw Exception('Failed to load history');
      }
    } catch (e) {
      throw Exception('Error loading history: $e');
    }
  }
  
  /// Get statistics
  Future<Map<String, dynamic>> getStatistics() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/statistics'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 10));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['data'];
      } else {
        throw Exception('Failed to load statistics');
      }
    } catch (e) {
      throw Exception('Error loading statistics: $e');
    }
  }
  
  /// Get available classes
  Future<List<String>> getClasses() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/classes'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return List<String>.from(data['classes']);
      } else {
        throw Exception('Failed to load classes');
      }
    } catch (e) {
      throw Exception('Error loading classes: $e');
    }
  }
  
  /// Delete prediction by ID
  Future<bool> deletePrediction(int predictionId) async {
    try {
      final response = await http.delete(
        Uri.parse('$baseUrl/history/$predictionId'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 5));
      
      return response.statusCode == 200;
    } catch (e) {
      print('Error deleting prediction: $e');
      return false;
    }
  }
}
```

## Model Classes

Buat file `lib/models/prediction_result.dart`:

```dart
class PredictionResult {
  final bool success;
  final int id;
  final String filename;
  final PredictionDetails svm;
  final PredictionDetails knn;
  final ManhattanPrediction? manhattan;
  final String timestamp;
  
  PredictionResult({
    required this.success,
    required this.id,
    required this.filename,
    required this.svm,
    required this.knn,
    this.manhattan,
    required this.timestamp,
  });
  
  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      success: json['success'],
      id: json['id'],
      filename: json['filename'],
      svm: PredictionDetails.fromJson(json['predictions']['svm']),
      knn: PredictionDetails.fromJson(json['predictions']['knn']),
      manhattan: json['predictions']['manhattan'] != null
          ? ManhattanPrediction.fromJson(json['predictions']['manhattan'])
          : null,
      timestamp: json['timestamp'],
    );
  }
}

class PredictionDetails {
  final String className;
  final double confidence;
  
  PredictionDetails({
    required this.className,
    required this.confidence,
  });
  
  factory PredictionDetails.fromJson(Map<String, dynamic> json) {
    return PredictionDetails(
      className: json['class'],
      confidence: json['confidence'].toDouble(),
    );
  }
}

class ManhattanPrediction {
  final String className;
  final Map<String, double> distances;
  
  ManhattanPrediction({
    required this.className,
    required this.distances,
  });
  
  factory ManhattanPrediction.fromJson(Map<String, dynamic> json) {
    return ManhattanPrediction(
      className: json['class'],
      distances: Map<String, double>.from(
        json['distances'].map((key, value) => MapEntry(key, value.toDouble()))
      ),
    );
  }
}
```

## Usage Example

Buat file `lib/screens/prediction_screen.dart`:

```dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/flower_api_service.dart';
import '../models/prediction_result.dart';

class PredictionScreen extends StatefulWidget {
  @override
  _PredictionScreenState createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final FlowerApiService _apiService = FlowerApiService();
  final ImagePicker _picker = ImagePicker();
  
  File? _selectedImage;
  PredictionResult? _result;
  bool _isLoading = false;
  String? _error;
  
  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(source: source);
      
      if (image != null) {
        setState(() {
          _selectedImage = File(image.path);
          _result = null;
          _error = null;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Failed to pick image: $e';
      });
    }
  }
  
  Future<void> _predictFlower() async {
    if (_selectedImage == null) return;
    
    setState(() {
      _isLoading = true;
      _error = null;
    });
    
    try {
      final response = await _apiService.predictFlower(_selectedImage!);
      final result = PredictionResult.fromJson(response);
      
      setState(() {
        _result = result;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Prediction failed: $e';
        _isLoading = false;
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flower Classification'),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image picker buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: Icon(Icons.camera_alt),
                    label: Text('Camera'),
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: Icon(Icons.photo_library),
                    label: Text('Gallery'),
                  ),
                ),
              ],
            ),
            
            SizedBox(height: 20),
            
            // Selected image preview
            if (_selectedImage != null) ...[
              Container(
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.file(_selectedImage!, fit: BoxFit.cover),
                ),
              ),
              
              SizedBox(height: 16),
              
              // Predict button
              ElevatedButton(
                onPressed: _isLoading ? null : _predictFlower,
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 16),
                ),
                child: _isLoading
                    ? CircularProgressIndicator(color: Colors.white)
                    : Text('Predict Flower', style: TextStyle(fontSize: 18)),
              ),
            ],
            
            SizedBox(height: 20),
            
            // Error message
            if (_error != null)
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red.shade100,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(_error!, style: TextStyle(color: Colors.red.shade900)),
              ),
            
            // Prediction results
            if (_result != null) ...[
              Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '🌸 Prediction Results',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Divider(),
                      
                      _buildPredictionRow(
                        'SVM',
                        _result!.svm.className,
                        _result!.svm.confidence,
                      ),
                      
                      SizedBox(height: 12),
                      
                      _buildPredictionRow(
                        'KNN',
                        _result!.knn.className,
                        _result!.knn.confidence,
                      ),
                      
                      if (_result!.manhattan != null) ...[
                        SizedBox(height: 12),
                        Text(
                          'Manhattan: ${_result!.manhattan!.className}',
                          style: TextStyle(fontSize: 16),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildPredictionRow(String method, String className, double confidence) {
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '$method: $className',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 4),
              LinearProgressIndicator(
                value: confidence / 100,
                backgroundColor: Colors.grey.shade300,
                valueColor: AlwaysStoppedAnimation<Color>(
                  confidence > 80 ? Colors.green : Colors.orange,
                ),
              ),
            ],
          ),
        ),
        SizedBox(width: 16),
        Text(
          '${confidence.toStringAsFixed(1)}%',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: confidence > 80 ? Colors.green : Colors.orange,
          ),
        ),
      ],
    );
  }
}
```

## Network Configuration

### Android (`android/app/src/main/AndroidManifest.xml`):

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    
    <application
        android:usesCleartextTraffic="true"
        ...>
        ...
    </application>
</manifest>
```

### iOS (`ios/Runner/Info.plist`):

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to capture flower images</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select flower images</string>
```

## Testing

```dart
// Test connection
void testConnection() async {
  final apiService = FlowerApiService();
  final isHealthy = await apiService.isServerHealthy();
  
  if (isHealthy) {
    print('✅ Server is healthy and ready!');
  } else {
    print('❌ Server is not available');
  }
}
```

## Notes

1. **Emulator**: Gunakan IP `10.0.2.2` untuk Android Emulator
2. **Physical Device**: Pastikan device dan server di network yang sama, gunakan IP komputer (cek dengan `ipconfig` di Windows atau `ifconfig` di Linux/Mac)
3. **CORS**: Sudah di-handle di server dengan `allow_origins=["*"]`
4. **Timeout**: Set timeout sesuai kebutuhan (default: 30s untuk prediction)
5. **Error Handling**: Selalu handle connection errors dan timeout

## Deployment

Untuk production:
1. Deploy server ke cloud (AWS, Azure, GCP, etc.)
2. Ganti `baseUrl` dengan URL production
3. Tambahkan authentication (JWT/API Key)
4. Enable HTTPS
5. Set proper CORS origins
