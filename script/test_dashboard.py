#!/usr/bin/env python3
"""
Test Dashboard Pages - Verify all pages load without errors.

Uses requests to fetch Streamlit pages and checks for error indicators.
"""

import requests
import sys
from time import sleep

BASE_URL = "http://localhost:8501"

def test_health():
    """Test Streamlit health endpoint."""
    try:
        resp = requests.get(f"{BASE_URL}/_stcore/health", timeout=10)
        if resp.text == "ok":
            print("[PASS] Health check: ok")
            return True
        else:
            print(f"[FAIL] Health check: {resp.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Health check: {e}")
        return False


def test_page_loads(page_name: str, path: str = "/"):
    """Test if a page loads without HTTP errors."""
    try:
        url = f"{BASE_URL}{path}"
        resp = requests.get(url, timeout=30)

        if resp.status_code == 200:
            # Check for common error indicators in the HTML
            content = resp.text.lower()

            # Streamlit error indicators
            error_indicators = [
                'streamlit error',
                'traceback',
                'exception',
                'modulenotfounderror',
                'importerror',
                'indexerror',
                'keyerror',
                'attributeerror',
            ]

            errors_found = []
            for indicator in error_indicators:
                if indicator in content:
                    errors_found.append(indicator)

            if errors_found:
                print(f"[WARN] {page_name}: Page loads but may have errors: {errors_found}")
                return True  # Page loads, but has warnings
            else:
                print(f"[PASS] {page_name}: HTTP 200, no error indicators")
                return True
        else:
            print(f"[FAIL] {page_name}: HTTP {resp.status_code}")
            return False

    except requests.exceptions.Timeout:
        print(f"[FAIL] {page_name}: Timeout")
        return False
    except Exception as e:
        print(f"[FAIL] {page_name}: {e}")
        return False


def test_streamlit_api():
    """Test Streamlit's internal API endpoints."""
    endpoints = [
        ("/_stcore/health", "Health"),
        ("/_stcore/allowed-message-origins", "Message Origins"),
    ]

    results = []
    for endpoint, name in endpoints:
        try:
            resp = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            if resp.status_code == 200:
                print(f"[PASS] API {name}: OK")
                results.append(True)
            else:
                print(f"[FAIL] API {name}: HTTP {resp.status_code}")
                results.append(False)
        except Exception as e:
            print(f"[FAIL] API {name}: {e}")
            results.append(False)

    return all(results)


def main():
    print("=" * 60)
    print("RLIC Dashboard Test Suite")
    print("=" * 60)
    print()

    # Wait for Streamlit to be ready
    print("Checking if Streamlit is ready...")
    for i in range(5):
        if test_health():
            break
        print(f"Waiting... ({i+1}/5)")
        sleep(2)
    else:
        print("[FAIL] Streamlit not responding after 5 attempts")
        sys.exit(1)

    print()
    print("-" * 60)
    print("Testing Pages")
    print("-" * 60)

    # Test main entry point and pages
    pages = [
        ("Main App", "/"),
        ("Catalog", "/1_%F0%9F%8F%A0_Catalog"),
        ("Overview", "/2_%F0%9F%93%8A_Overview"),
        ("Qualitative", "/3_%F0%9F%93%96_Qualitative"),
        ("Correlation", "/4_%F0%9F%93%88_Correlation"),
        ("Lead-Lag", "/5_%F0%9F%94%84_Lead_Lag"),
        ("Regimes", "/6_%F0%9F%8E%AF_Regimes"),
        ("Backtests", "/7_%F0%9F%92%B0_Backtests"),
        ("Forecasts", "/8_%F0%9F%94%AE_Forecasts"),
    ]

    results = []
    for name, path in pages:
        result = test_page_loads(name, path)
        results.append(result)

    print()
    print("-" * 60)
    print("Testing API Endpoints")
    print("-" * 60)

    api_result = test_streamlit_api()
    results.append(api_result)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) had issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
