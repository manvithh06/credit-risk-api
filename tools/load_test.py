"""
tools/load_test.py — Load test the /predict endpoint.

Sends 100 requests and reports p50, p95, p99 latency.
Requirement: p95 < 200ms on a free-tier host.

Usage:
  python tools/load_test.py --url http://localhost:8000
  python tools/load_test.py --url https://your-deployed-api.com
"""

import requests
import time
import numpy as np
import argparse
import json

# Sample payload — realistic credit applicant
SAMPLE_PAYLOAD = {
    "age": 35,
    "income": 65000,
    "loan_amount": 15000,
    "credit_score": 680,
    "employment_years": 5,
    "debt_to_income": 0.35,
    "num_credit_lines": 4,
    "payment_history": 0.9,
    "loan_to_value": 0.7,
    "num_late_payments": 1
}
session = requests.Session() 

def run_load_test(base_url: str, n_requests: int = 100):
    url = f"{base_url}/predict"
    latencies = []
    errors = 0
    
    print(f"Load testing {url}")
    print(f"Sending {n_requests} requests...\n")
    
    # Warmup — first request is always slower (cold start)
    try:
        response = session.post(url, json=SAMPLE_PAYLOAD, timeout=10)
    except:
        pass
    
    for i in range(n_requests):
        start = time.time()
        try:
            response = session.post(url, json=SAMPLE_PAYLOAD, timeout=10)
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                latencies.append(latency_ms)
            else:
                errors += 1
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            errors += 1
            print(f"Request failed: {e}")
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_requests} requests sent...")
    
    if not latencies:
        print("❌ All requests failed. Is the API running?")
        return
    
    # Results
    print(f"\n{'='*45}")
    print("LOAD TEST RESULTS")
    print(f"{'='*45}")
    print(f"Total requests:  {n_requests}")
    print(f"Successful:      {len(latencies)}")
    print(f"Errors:          {errors}")
    print(f"\nLatency (ms):")
    print(f"  Min:    {min(latencies):.1f}ms")
    print(f"  Mean:   {np.mean(latencies):.1f}ms")
    print(f"  Median: {np.median(latencies):.1f}ms")
    print(f"  p95:    {np.percentile(latencies, 95):.1f}ms  ← must be <200ms")
    print(f"  p99:    {np.percentile(latencies, 99):.1f}ms")
    print(f"  Max:    {max(latencies):.1f}ms")
    
    p95 = np.percentile(latencies, 95)
    if p95 < 200:
        print(f"\n✅ PASSED — p95 latency {p95:.1f}ms < 200ms requirement")
    else:
        print(f"\n❌ FAILED — p95 latency {p95:.1f}ms exceeds 200ms requirement")
    
    # Save results
    results = {
        "url": url,
        "n_requests": n_requests,
        "successful": len(latencies),
        "errors": errors,
        "p50_ms": round(np.median(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "p99_ms": round(np.percentile(latencies, 99), 2),
        "mean_ms": round(np.mean(latencies), 2),
        "passed_sla": bool(p95 < 200)
    }
    
    with open('tools/load_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: tools/load_test_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000',
                        help='Base URL of the API')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of requests')
    args = parser.parse_args()
    run_load_test(args.url, args.n)
