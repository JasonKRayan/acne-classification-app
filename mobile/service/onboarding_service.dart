import 'package:shared_preferences/shared_preferences.dart';

class OnboardingService {
  static const String _onboardingKey = 'onboarding_complete';
  static const String _firstLaunchKey = 'first_launch_date';
  static const String _onboardingVersionKey = 'onboarding_version';

  // Current onboarding version - increment when you update onboarding flow
  static const int currentVersion = 1;

  /// Check if user has completed onboarding
  static Future<bool> hasCompletedOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    final completed = prefs.getBool(_onboardingKey) ?? false;
    final version = prefs.getInt(_onboardingVersionKey) ?? 0;

    // Show onboarding again if version changed
    return completed && version >= currentVersion;
  }

  /// Mark onboarding as complete
  static Future<void> completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_onboardingKey, true);
    await prefs.setInt(_onboardingVersionKey, currentVersion);

    // Save first launch date if not already saved
    if (!prefs.containsKey(_firstLaunchKey)) {
      await prefs.setString(_firstLaunchKey, DateTime.now().toIso8601String());
    }
  }

  /// Reset onboarding (for testing or user preference)
  static Future<void> resetOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_onboardingKey, false);
    await prefs.remove(_onboardingVersionKey);
  }

  /// Get first launch date
  static Future<DateTime?> getFirstLaunchDate() async {
    final prefs = await SharedPreferences.getInstance();
    final dateString = prefs.getString(_firstLaunchKey);
    if (dateString != null) {
      return DateTime.parse(dateString);
    }
    return null;
  }

  /// Check if this is first app launch
  static Future<bool> isFirstLaunch() async {
    final prefs = await SharedPreferences.getInstance();
    return !prefs.containsKey(_firstLaunchKey);
  }

  /// Get onboarding completion status details
  static Future<Map<String, dynamic>> getOnboardingStatus() async {
    final prefs = await SharedPreferences.getInstance();
    final completed = prefs.getBool(_onboardingKey) ?? false;
    final version = prefs.getInt(_onboardingVersionKey) ?? 0;
    final firstLaunch = prefs.getString(_firstLaunchKey);

    return {
      'completed': completed,
      'version': version,
      'current_version': currentVersion,
      'needs_update': version < currentVersion,
      'first_launch': firstLaunch,
      'is_first_launch': firstLaunch == null,
    };
  }
}