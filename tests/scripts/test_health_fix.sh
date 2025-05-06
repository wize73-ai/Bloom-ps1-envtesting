#!/bin/bash
# Test script for health endpoint fix

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}Testing health endpoint fix${NC}"
echo -e "${BLUE}=============================${NC}"

# Start the server in the background
echo -e "${BLUE}Starting server...${NC}"
python -m app.main &
SERVER_PID=$!

# Wait for server to start
echo -e "${BLUE}Waiting for server to start...${NC}"
sleep 10

# Check if the server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo -e "${RED}Server failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}Server started with PID $SERVER_PID${NC}"

# Test health endpoint
echo -e "${BLUE}Testing /health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)

# Check if database status is healthy
if echo "$HEALTH_RESPONSE" | grep -q '"database":"healthy"'; then
    echo -e "${GREEN}SUCCESS: Database health check reports 'healthy'${NC}"
else
    echo -e "${RED}FAIL: Database health check does not report 'healthy'${NC}"
    echo "Response:"
    echo "$HEALTH_RESPONSE" | python -m json.tool
fi

# Print the full response
echo -e "${BLUE}Full health response:${NC}"
echo "$HEALTH_RESPONSE" | python -m json.tool

# Test database health endpoint
echo -e "${BLUE}Testing /health/database endpoint...${NC}"
DB_HEALTH_RESPONSE=$(curl -s http://localhost:8000/health/database)

# Print the full response
echo -e "${BLUE}Full database health response:${NC}"
echo "$DB_HEALTH_RESPONSE" | python -m json.tool

# Test readiness endpoint
echo -e "${BLUE}Testing /readiness endpoint...${NC}"
READINESS_RESPONSE=$(curl -s http://localhost:8000/readiness)

# Print the full response
echo -e "${BLUE}Full readiness response:${NC}"
echo "$READINESS_RESPONSE" | python -m json.tool

# Stop the server
echo -e "${BLUE}Stopping server...${NC}"
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo -e "${GREEN}Test completed!${NC}"