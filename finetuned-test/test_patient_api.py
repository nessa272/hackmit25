"""
Test file for patient API endpoints using the NSTEMI test case from single_case_test.py.
Tests both POST prediction endpoint and GET utility endpoints.
"""

import requests
import json
import time

# API endpoints (from your current Modal serve output)
BASE_URL = "https://nessa272--predict-los-dev.modal.run"
HEALTH_CHECK_URL = "https://nessa272--health-check-dev.modal.run"
MODEL_INFO_URL = "https://nessa272--models-dev.modal.run"

# Test case data from single_case_test.py
NSTEMI_TEST_CASE = {
    "age": 67,
    "gender": "Male",
    "diagnosis": "NSTEMI (I21.4), status post PCI with drug-eluting stent to mid-LAD",
    "severity": "Moderate severity",
    "mortality_risk": "Moderate risk",
    "admission_type": "emergency",
    "expected_disposition": "home",
    "additional_info": "Race: Unknown, Ethnicity: Unknown"
}

# Expected actual LOS for comparison
ACTUAL_LOS = 6

def test_health_check():
    """Test the health check GET endpoint."""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info GET endpoint."""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(MODEL_INFO_URL, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Model info retrieved successfully")
            return True
        else:
            print("❌ Model info failed")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_prediction_endpoint():
    """Test the main prediction POST endpoint with NSTEMI case."""
    print("\n🔍 Testing prediction endpoint with NSTEMI case...")
    print(f"📋 Test Case: {NSTEMI_TEST_CASE['age']} year old {NSTEMI_TEST_CASE['gender']}")
    print(f"📋 Diagnosis: {NSTEMI_TEST_CASE['diagnosis']}")
    print(f"📋 Actual LOS: {ACTUAL_LOS} days")
    
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        
        print("\n⏳ Sending prediction request...")
        start_time = time.time()
        
        response = requests.post(
            BASE_URL, 
            json=NSTEMI_TEST_CASE, 
            headers=headers,
            timeout=300  # 5 minutes timeout for model inference
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful!")
            print("=" * 60)
            print("🤖 PREDICTION RESULTS")
            print("=" * 60)
            print(f"📈 Predicted LOS: {result.get('predicted_los_days', 'N/A')} days")
            print(f"📊 Actual LOS: {ACTUAL_LOS} days")
            
            predicted_los = result.get('predicted_los_days', 0)
            error = abs(predicted_los - ACTUAL_LOS)
            print(f"📉 Prediction Error: {error} days")
            
            print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
            print(f"👤 Patient Summary: {result.get('patient_summary', 'N/A')}")
            
            print("\n🧠 Medical Reasoning:")
            print("-" * 40)
            reasoning = result.get('reasoning', 'No reasoning provided')
            # Format reasoning for better readability
            if len(reasoning) > 100:
                words = reasoning.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(' '.join(current_line)) > 80:
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                for line in lines:
                    print(line)
            else:
                print(reasoning)
            
            print("\n" + "=" * 60)
            
            # Evaluate prediction quality
            if error <= 1:
                print("🎉 Excellent prediction (≤1 day error)")
            elif error <= 2:
                print("👍 Good prediction (≤2 days error)")
            elif error <= 3:
                print("👌 Acceptable prediction (≤3 days error)")
            else:
                print("⚠️  High error prediction (>3 days error)")
            
            return True, result
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (model inference took too long)")
        return False, None
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False, None

def test_invalid_input():
    """Test the prediction endpoint with invalid input."""
    print("\n🔍 Testing prediction endpoint with invalid input...")
    
    invalid_cases = [
        # Missing required fields
        {"age": 67},
        # Invalid data types
        {"age": "not_a_number", "gender": "Male", "diagnosis": "Test"},
        # Empty request
        {}
    ]
    
    for i, invalid_case in enumerate(invalid_cases, 1):
        print(f"\n📋 Invalid Test Case {i}: {invalid_case}")
        try:
            response = requests.post(
                BASE_URL, 
                json=invalid_case, 
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code != 200:
                print("✅ Correctly rejected invalid input")
            else:
                print("⚠️  API accepted invalid input (might want to add validation)")
        except Exception as e:
            print(f"❌ Error testing invalid input: {e}")

def run_all_tests():
    """Run all API tests."""
    print("🚀 Starting Patient API Tests")
    print("=" * 60)
    
    # Test GET endpoints first (faster)
    health_ok = test_health_check()
    model_info_ok = test_model_info()
    
    # Test main prediction endpoint
    prediction_ok, prediction_result = test_prediction_endpoint()
    
    # Test error handling
    test_invalid_input()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Model Info: {'✅ PASS' if model_info_ok else '❌ FAIL'}")
    print(f"Prediction: {'✅ PASS' if prediction_ok else '❌ FAIL'}")
    
    if prediction_ok and prediction_result:
        print(f"\n🎯 Final Prediction: {prediction_result.get('predicted_los_days')} days")
        print(f"📊 Actual LOS: {ACTUAL_LOS} days")
        error = abs(prediction_result.get('predicted_los_days', 0) - ACTUAL_LOS)
        print(f"📉 Error: {error} days")
    
    overall_success = health_ok and model_info_ok and prediction_ok
    print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    print("Patient API Test Suite")
    print("Make sure your Modal API is running before starting tests!")
    print("Update the BASE_URL, HEALTH_CHECK_URL, and MODEL_INFO_URL with your actual endpoints.")
    print()
    
    # Wait a moment for user to read
    input("Press Enter to start tests...")
    
    success = run_all_tests()
    exit(0 if success else 1)
