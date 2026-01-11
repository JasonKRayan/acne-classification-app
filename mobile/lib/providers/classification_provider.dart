import 'dart:io';
import 'package:flutter/foundation.dart';
import '../models/classification_result.dart';
import '../models/health_status.dart';
import '../models/model_info.dart';
import '../services/api_service.dart';

class ClassificationProvider with ChangeNotifier {
  final ApiService _apiService = ApiService();

  // State
  bool _isLoading = false;
  bool _isConnected = false;
  String? _error;
  ClassificationResult? _currentResult;
  HealthStatus? _healthStatus;
  ModelInfo? _modelInfo;
  File? _selectedImage;

  // Getters
  bool get isLoading => _isLoading;
  bool get isConnected => _isConnected;
  String? get error => _error;
  ClassificationResult? get currentResult => _currentResult;
  HealthStatus? get healthStatus => _healthStatus;
  ModelInfo? get modelInfo => _modelInfo;
  File? get selectedImage => _selectedImage;

  // Set selected image
  void setSelectedImage(File? image) {
    _selectedImage = image;
    _error = null;
    notifyListeners();
  }

  // Clear result
  void clearResult() {
    _currentResult = null;
    _selectedImage = null;
    _error = null;
    notifyListeners();
  }

  // Check API connection
  Future<void> checkConnection() async {
    try {
      _isLoading = true;
      notifyListeners();

      _isConnected = await _apiService.testConnection();

      if (_isConnected) {
        await checkHealth();
      }
    } catch (e) {
      _isConnected = false;
      _error = 'Failed to connect to server';
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  // Check health status
  Future<void> checkHealth() async {
    try {
      _healthStatus = await _apiService.checkHealth();
      _isConnected = _healthStatus?.isHealthy ?? false;
      _error = null;
    } catch (e) {
      _error = e.toString();
      _isConnected = false;
    }
    notifyListeners();
  }

  // Get model information
  Future<void> fetchModelInfo() async {
    try {
      _isLoading = true;
      _error = null;
      notifyListeners();

      _modelInfo = await _apiService.getModelInfo();
    } catch (e) {
      _error = _getErrorMessage(e);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  // Classify image
  Future<void> classifyImage(File imageFile, {int topK = 3}) async {
    try {
      _isLoading = true;
      _error = null;
      _selectedImage = imageFile;
      notifyListeners();

      _currentResult = await _apiService.classifyImage(
        imageFile,
        topK: topK,
      );

      _error = null;
    } catch (e) {
      _error = _getErrorMessage(e);
      _currentResult = null;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  // Retry classification
  Future<void> retryClassification({int topK = 3}) async {
    if (_selectedImage != null) {
      await classifyImage(_selectedImage!, topK: topK);
    }
  }

  // Get user-friendly error message
  String _getErrorMessage(dynamic error) {
    if (error is ApiException) {
      return error.message;
    }
    return 'An unexpected error occurred';
  }

  // Clear error
  void clearError() {
    _error = null;
    notifyListeners();
  }

  // Get connection status message
  String get connectionStatusMessage {
    if (_isLoading) return 'Checking connection...';
    if (!_isConnected) return 'Not connected to server';
    if (_healthStatus?.modelLoaded == false) return 'Model not loaded';
    return 'Connected';
  }

  // Get connection status color
  int get connectionStatusColor {
    if (_isLoading) return 0xFFFF9800; // Orange
    if (!_isConnected) return 0xFFF44336; // Red
    if (_healthStatus?.modelLoaded == false) return 0xFFFF9800; // Orange
    return 0xFF4CAF50; // Green
  }
}