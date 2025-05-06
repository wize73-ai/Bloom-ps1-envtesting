#!/bin/bash
# Run SonarQube analysis for CasaLingua project

set -e  # Exit on error

# Configuration variables
SONAR_HOST_URL=${SONAR_HOST_URL:-"http://localhost:9000"}
SONAR_TOKEN=${SONAR_TOKEN:-""}
SONAR_PROJECT_KEY=${SONAR_PROJECT_KEY:-"casMay4"}

# Print header
echo "===================================="
echo "CasaLingua SonarQube Analysis Runner"
echo "===================================="
echo

# Check if SONAR_TOKEN is set
if [ -z "$SONAR_TOKEN" ]; then
    echo "Error: SONAR_TOKEN is not set. Please provide a valid SonarQube authentication token."
    echo "Usage: SONAR_TOKEN=your_token ./run_sonar_analysis.sh"
    exit 1
fi

# Create directories for reports if they don't exist
mkdir -p coverage-reports junit-reports

# Clean up any previous reports
echo "Cleaning up previous reports..."
rm -rf coverage-reports/* junit-reports/*

# Set Python environment variable to load env settings
export CASALINGUA_ENV=development

# Step 1: Run tests with coverage
echo "Running tests with coverage..."
python -m pytest tests app/tests \
    --cov=app \
    --cov-report=xml:coverage-reports/coverage.xml \
    --cov-report=html:coverage-reports/html \
    --junitxml=junit-reports/junit.xml

# Step 2: Run SonarQube analysis
echo "Running SonarQube analysis..."
sonar-scanner \
    -Dsonar.host.url=$SONAR_HOST_URL \
    -Dsonar.login=$SONAR_TOKEN \
    -Dsonar.projectKey=$SONAR_PROJECT_KEY

# Print summary
echo
echo "===================================="
echo "Analysis Complete!"
echo "Coverage reports are available in: coverage-reports/html/index.html"
echo "SonarQube Dashboard: $SONAR_HOST_URL/dashboard?id=$SONAR_PROJECT_KEY"
echo "===================================="