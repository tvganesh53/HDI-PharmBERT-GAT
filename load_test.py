"""
load_test.py — Locust load testing script for Phase F.

Run locally:
    locust -f load_test.py --host http://localhost:8001

Run headless (CI mode):
    locust -f load_test.py --headless --users 50 --spawn-rate 5 --run-time 60s --host http://localhost:8001

Open browser UI:
    http://localhost:8089
"""

from __future__ import annotations
import json
import os
import random
from locust import HttpUser, task, between, events

# ── Test data ─────────────────────────────────────────────────────────────────
SAMPLE_INPUTS = [
    "My payment failed and I need help immediately",
    "I want to cancel my subscription",
    "How do I reset my password?",
    "I was charged twice for the same order",
    "Can I upgrade to the premium plan?",
    "The app keeps crashing on my phone",
    "I need to update my billing address",
    "Where is my refund? It's been 2 weeks",
    "I can't log into my account",
    "Do you offer annual pricing?",
    "I found a bug in the dashboard feature",
    "How do I export my data?",
    "My account was locked out after 3 attempts",
    "I would like to request a new reporting feature",
    "What is the difference between basic and pro plans?",
]

# ── Get API key from env ──────────────────────────────────────────────────────
API_KEY = os.getenv("TEST_API_KEY", "")


# ── User behaviour ────────────────────────────────────────────────────────────
class ClassifierUser(HttpUser):
    """
    Simulates a real user hitting the classification API.
    Waits 1-3 seconds between requests (realistic usage pattern).
    """
    wait_time = between(1, 3)
    headers   = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(5)
    def classify_sync(self):
        """Most common task — sync classification."""
        text = random.choice(SAMPLE_INPUTS)
        with self.client.post(
            "/classify",
            data=json.dumps({"inputs": text}),
            headers=self.headers,
            catch_response=True,
            name="POST /classify",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()  # Rate limit is expected under load
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def classify_async(self):
        """Async classification — fire and forget."""
        text = random.choice(SAMPLE_INPUTS)
        with self.client.post(
            "/classify/async",
            data=json.dumps({"inputs": text}),
            headers=self.headers,
            catch_response=True,
            name="POST /classify/async",
        ) as response:
            if response.status_code in (200, 202, 429):
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def check_health(self):
        """Health check — no auth needed."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def check_stats(self):
        """Stats endpoint."""
        with self.client.get(
            "/stats",
            headers=self.headers,
            catch_response=True,
            name="GET /stats",
        ) as response:
            if response.status_code in (200, 401, 429):
                response.success()
            else:
                response.failure(f"Stats failed: {response.status_code}")

    @task(1)
    def batch_classify(self):
        """Batch classification with multiple texts."""
        texts = random.sample(SAMPLE_INPUTS, k=random.randint(2, 4))
        with self.client.post(
            "/classify",
            data=json.dumps({"inputs": texts}),
            headers=self.headers,
            catch_response=True,
            name="POST /classify (batch)",
        ) as response:
            if response.status_code in (200, 422, 429):
                response.success()
            else:
                response.failure(f"Batch classify failed: {response.status_code}")


# ── Spike test user (hits API much faster) ────────────────────────────────────
class SpikeUser(HttpUser):
    """Simulates a burst/spike of traffic with minimal wait time."""
    wait_time = between(0.1, 0.5)
    weight    = 1   # fewer spike users than normal users
    headers   = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task
    def rapid_classify(self):
        text = random.choice(SAMPLE_INPUTS)
        self.client.post(
            "/classify",
            data=json.dumps({"inputs": text}),
            headers=self.headers,
            name="POST /classify (spike)",
        )


# ── Event hooks for reporting ─────────────────────────────────────────────────
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "="*60)
    print("  Phase F Load Test Starting")
    print(f"  Target: {environment.host}")
    print(f"  API Key: {'SET' if API_KEY else 'NOT SET (auth will fail)'}")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    print("\n" + "="*60)
    print("  Phase F Load Test Complete")
    print(f"  Total requests:  {stats.num_requests}")
    print(f"  Failures:        {stats.num_failures}")
    print(f"  Avg response:    {stats.avg_response_time:.0f}ms")
    print(f"  Requests/sec:    {stats.current_rps:.1f}")
    print("="*60 + "\n")
