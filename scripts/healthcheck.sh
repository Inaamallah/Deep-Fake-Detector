#!/usr/bin/env bash
# scripts/healthcheck.sh  ← CREATE (also: mkdir scripts)
# End-to-end health check script that verifies all three services are up.
# Run this after `docker compose up` to confirm everything is working.
# Exit code 0 = all healthy, non-zero = something is wrong.

set -euo pipefail

API_URL="${1:-http://localhost:8000}"
NGINX_URL="${2:-http://localhost:80}"
API_KEY="${API_KEY:-dev-key-change-in-production}"

pass() { echo "  ✓  $1"; }
fail() { echo "  ✗  $1"; exit 1; }

echo ""
echo "═══════════════════════════════════════"
echo "  Deepfake Detector — Health Check"
echo "═══════════════════════════════════════"
echo ""

# 1. Direct API health check
echo "1. API server (direct)..."
response=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/api/v1/health")
[ "$response" = "200" ] && pass "FastAPI is healthy (HTTP 200)" \
                         || fail "FastAPI returned HTTP $response"

# 2. Nginx proxy health check
echo "2. Nginx proxy..."
response=$(curl -s -o /dev/null -w "%{http_code}" "${NGINX_URL}/api/v1/health")
[ "$response" = "200" ] && pass "Nginx proxy is healthy (HTTP 200)" \
                         || fail "Nginx returned HTTP $response"

# 3. Auth check — valid key should pass
echo "3. API authentication..."
response=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "X-API-Key: ${API_KEY}" "${API_URL}/api/v1/health")
[ "$response" = "200" ] && pass "Auth with valid key passes" \
                         || fail "Auth failed unexpectedly (HTTP $response)"

# 4. Auth check — missing key should return 403
echo "4. Reject missing API key..."
response=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/api/v1/jobs/url" \
  -X POST -H "Content-Type: application/json" -d '{"url":"https://example.com"}')
[ "$response" = "403" ] && pass "Missing key correctly rejected (HTTP 403)" \
                         || fail "Expected 403, got $response"

# 5. Prometheus metrics endpoint
echo "5. Prometheus metrics..."
response=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/metrics")
[ "$response" = "200" ] && pass "Metrics endpoint accessible" \
                         || fail "Metrics endpoint returned HTTP $response"

# 6. React frontend served by Nginx
echo "6. React frontend..."
response=$(curl -s -o /dev/null -w "%{http_code}" "${NGINX_URL}/")
[ "$response" = "200" ] && pass "Frontend index.html served by Nginx" \
                         || fail "Frontend returned HTTP $response"

# 7. Submit a test URL job and verify it gets queued
echo "7. Job submission..."
job_response=$(curl -s -X POST "${API_URL}/api/v1/jobs/url" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}')
job_status=$(echo "$job_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
[ "$job_status" = "PENDING" ] && pass "Job submission returns PENDING status" \
                               || fail "Unexpected job response: $job_response"

echo ""
echo "═══════════════════════════════════════"
echo "  All checks passed ✓"
echo "═══════════════════════════════════════"
echo ""