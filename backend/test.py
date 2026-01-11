"""
Simple test client for the Acne Classification API
"""
import requests
import json
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{API_BASE_URL}/api/v1/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_model_info():
    """Test the model info endpoint"""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{API_BASE_URL}/api/v1/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_classify_image(image_path: str, top_k: int = 3):
    """Test the classification endpoint"""
    print(f"\n=== Testing Classification with {image_path} ===")

    if not Path(image_path).exists():
        print(f"Error: Image file not found at {image_path}")
        return False

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        params = {"top_k": top_k}

        response = requests.post(
            f"{API_BASE_URL}/api/v1/classify",
            files=files,
            params=params
        )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Classification Successful!")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nTop {top_k} Predictions:")
        for pred in result['all_predictions']:
            print(f"  - {pred['class_name']}: {pred['confidence_percentage']:.2f}%")

        if result.get('recommendations'):
            print(f"\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")

        print(f"\nProcessing Time: {result['processing_time_ms']:.2f}ms")
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_invalid_file():
    """Test with invalid file type"""
    print("\n=== Testing Invalid File Type ===")

    # Create a temporary text file
    temp_file = "temp_test.txt"
    with open(temp_file, "w") as f:
        f.write("This is not an image")

    with open(temp_file, "rb") as f:
        files = {"file": (temp_file, f, "text/plain")}
        response = requests.post(f"{API_BASE_URL}/api/v1/classify", files=files)

    # Clean up
    Path(temp_file).unlink()

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    return response.status_code == 400


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("ACNE CLASSIFICATION API TEST SUITE")
    print("=" * 60)

    results = {
        "Health Check": test_health_check(),
        "Model Info": test_model_info(),
        "Invalid File": test_invalid_file(),
    }

    # Test with sample image if provided
    sample_image = "sample_acne.jpg"  # Replace with actual image path
    if Path(sample_image).exists():
        results["Image Classification"] = test_classify_image(sample_image)
    else:
        print(f"\n⚠️  Skipping image classification test (no image found at {sample_image})")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()