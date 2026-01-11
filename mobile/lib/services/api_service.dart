import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/classification_result.dart';
import '../models/health_status.dart';
import '../models/model_info.dart';

class ApiService {
  // Update this with your backend URL
  static const String baseUrl = 'http://localhost:8000';
  // For Android emulator: 'http://10.0.2.2:8000'
  // For iOS simulator: 'http://localhost:8000'
  // For real device: 'http://YOUR_COMPUTER_IP:8000'

  static const Duration timeout = Duration(seconds: 30);

  // Health Check
  Future<HealthStatus> checkHealth() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/api/v1/health'),
          )
          .timeout(timeout);

      if (response.statusCode == 200) {
        return HealthStatus.fromJson(json.decode(response.body));
      } else {
        throw ApiException(
          'Health check failed',
          statusCode: response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException('No internet connection');
    } on http.ClientException {
      throw ApiException('Cannot connect to server');
    } on TimeoutException {
      throw ApiException('Request timeout');
    } catch (e) {
      throw ApiException('Unexpected error: ${e.toString()}');
    }
  }

  // Get Model Info
  Future<ModelInfo> getModelInfo() async {
    try {
      final response = await http
          .get(
            Uri.parse('$baseUrl/api/v1/model/info'),
          )
          .timeout(timeout);

      if (response.statusCode == 200) {
        return ModelInfo.fromJson(json.decode(response.body));
      } else {
        throw ApiException(
          'Failed to get model info',
          statusCode: response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException('No internet connection');
    } on http.ClientException {
      throw ApiException('Cannot connect to server');
    } on TimeoutException {
      throw ApiException('Request timeout');
    } catch (e) {
      throw ApiException('Unexpected error: ${e.toString()}');
    }
  }

  // Classify Image
  Future<ClassificationResult> classifyImage(
    File imageFile, {
    int topK = 3,
  }) async {
    try {
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/api/v1/classify?top_k=$topK'),
      );

      // Add image file
      var stream = http.ByteStream(imageFile.openRead());
      var length = await imageFile.length();
      var multipartFile = http.MultipartFile(
        'file',
        stream,
        length,
        filename: imageFile.path.split('/').last,
      );
      request.files.add(multipartFile);

      // Send request
      var streamedResponse = await request.send().timeout(timeout);
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return ClassificationResult.fromJson(data);
      } else if (response.statusCode == 400) {
        final error = json.decode(response.body);
        throw ApiException(
          error['detail'] ?? 'Invalid image',
          statusCode: 400,
        );
      } else if (response.statusCode == 500) {
        throw ApiException(
          'Server error during classification',
          statusCode: 500,
        );
      } else {
        throw ApiException(
          'Classification failed',
          statusCode: response.statusCode,
        );
      }
    } on SocketException {
      throw ApiException('No internet connection');
    } on http.ClientException {
      throw ApiException('Cannot connect to server');
    } on TimeoutException {
      throw ApiException('Request timeout - image may be too large');
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Unexpected error: ${e.toString()}');
    }
  }

  // Test Connection
  Future<bool> testConnection() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

// Custom Exception Class
class ApiException implements Exception {
  final String message;
  final int? statusCode;

  ApiException(this.message, {this.statusCode});

  @override
  String toString() => message;
}

// Timeout Exception
class TimeoutException implements Exception {
  final String message;
  TimeoutException([this.message = 'Request timeout']);

  @override
  String toString() => message;
}