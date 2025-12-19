class ClassificationResult {
  final bool success;
  final String predictedClass;
  final double confidence;
  final List<PredictionResult> allPredictions;
  final List<String> recommendations;
  final DateTime timestamp;
  final double? processingTimeMs;

  ClassificationResult({
    required this.success,
    required this.predictedClass,
    required this.confidence,
    required this.allPredictions,
    required this.recommendations,
    required this.timestamp,
    this.processingTimeMs,
  });

  factory ClassificationResult.fromJson(Map<String, dynamic> json) {
    return ClassificationResult(
      success: json['success'] ?? true,
      predictedClass: json['predicted_class'],
      confidence: (json['confidence'] as num).toDouble(),
      allPredictions: (json['all_predictions'] as List)
          .map((p) => PredictionResult.fromJson(p))
          .toList(),
      recommendations: (json['recommendations'] as List)
          .map((r) => r.toString())
          .toList(),
      timestamp: DateTime.parse(json['timestamp']),
      processingTimeMs: json['processing_time_ms']?.toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'success': success,
      'predicted_class': predictedClass,
      'confidence': confidence,
      'all_predictions': allPredictions.map((p) => p.toJson()).toList(),
      'recommendations': recommendations,
      'timestamp': timestamp.toIso8601String(),
      'processing_time_ms': processingTimeMs,
    };
  }

  // Get confidence as percentage
  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';

  // Get severity level based on acne type
  String get severityLevel {
    switch (predictedClass.toLowerCase()) {
      case 'blackheads':
      case 'whiteheads':
        return 'Mild';
      case 'papules':
      case 'pustules':
      case 'dark spot':
        return 'Moderate';
      case 'nodules':
        return 'Severe';
      default:
        return 'Unknown';
    }
  }

  // Get color for severity
  int get severityColor {
    switch (severityLevel) {
      case 'Mild':
        return 0xFF4CAF50; // Green
      case 'Moderate':
        return 0xFFFF9800; // Orange
      case 'Severe':
        return 0xFFF44336; // Red
      default:
        return 0xFF9E9E9E; // Grey
    }
  }
}

class PredictionResult {
  final String className;
  final double confidence;
  final double confidencePercentage;

  PredictionResult({
    required this.className,
    required this.confidence,
    required this.confidencePercentage,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      className: json['class_name'],
      confidence: (json['confidence'] as num).toDouble(),
      confidencePercentage: (json['confidence_percentage'] as num).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'class_name': className,
      'confidence': confidence,
      'confidence_percentage': confidencePercentage,
    };
  }

  String get formattedPercentage => '${confidencePercentage.toStringAsFixed(1)}%';
}

class HealthStatus {
  final String status;
  final bool modelLoaded;
  final DateTime timestamp;

  HealthStatus({
    required this.status,
    required this.modelLoaded,
    required this.timestamp,
  });

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      status: json['status'],
      modelLoaded: json['model_loaded'],
      timestamp: DateTime.parse(json['timestamp']),
    );
  }

  bool get isHealthy => status == 'healthy' && modelLoaded;
}

class ModelInfo {
  final String modelName;
  final String version;
  final List<String> classes;
  final List<int> inputSize;
  final int totalClasses;

  ModelInfo({
    required this.modelName,
    required this.version,
    required this.classes,
    required this.inputSize,
    required this.totalClasses,
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      modelName: json['model_name'],
      version: json['version'],
      classes: (json['classes'] as List).map((c) => c.toString()).toList(),
      inputSize: (json['input_size'] as List).map((i) => i as int).toList(),
      totalClasses: json['total_classes'],
    );
  }
}