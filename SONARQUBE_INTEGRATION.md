# SonarQube Integration Guide

This document explains how to use SonarQube with the CasaLingua project to get comprehensive code quality reports, including test coverage metrics.

## Prerequisites

1. **SonarQube Server** - Either a self-hosted instance or a SonarCloud account
2. **Python 3.9+** - Compatible with the project's requirements
3. **Project Dependencies** - Install all project dependencies with the added SonarQube components

## Setup

### 1. Install Dependencies

The project's `requirements.txt` has been updated to include SonarQube-related dependencies. Install them with:

```bash
pip install -r requirements.txt
```

### 2. SonarQube Server

If you don't have a SonarQube server running, you can start one using Docker:

```bash
docker run -d --name sonarqube -p 9000:9000 sonarqube:latest
```

Once started, access the SonarQube dashboard at http://localhost:9000 (default credentials are admin/admin).

### 3. Create a Project and Token

1. Log in to your SonarQube instance
2. Create a new project with key `casMay4` (or update the key in `sonar-project.properties`)
3. Generate a new token for authentication (User > My Account > Security > Generate Tokens)

## Running Analysis

### Using the Provided Script

We've created a script that automates running tests with coverage and uploading results to SonarQube:

```bash
# Set your SonarQube token (replace with your actual token)
export SONAR_TOKEN=your_token_here

# Run the analysis script
./run_sonar_analysis.sh
```

By default, the script uses:
- SonarQube URL: http://localhost:9000 (override with SONAR_HOST_URL)
- Project key: casMay4 (override with SONAR_PROJECT_KEY)

### Manual Analysis

If you prefer to run the steps manually:

1. **Run tests with coverage**:
   ```bash
   python -m pytest tests app/tests \
       --cov=app \
       --cov-report=xml:coverage-reports/coverage.xml \
       --cov-report=html:coverage-reports/html \
       --junitxml=junit-reports/junit.xml
   ```

2. **Run SonarQube scanner**:
   ```bash
   sonar-scanner \
       -Dsonar.host.url=http://localhost:9000 \
       -Dsonar.login=your_token_here \
       -Dsonar.projectKey=casMay4
   ```

## Understanding Results

After running the analysis, you can view the results in your SonarQube dashboard:

1. Go to http://localhost:9000/dashboard?id=casMay4
2. Review code quality metrics:
   - Test coverage percentage
   - Code duplication
   - Code smells
   - Bugs
   - Security vulnerabilities
   - Technical debt

## Coverage Reports

The analysis also generates local coverage reports:

- HTML report: `coverage-reports/html/index.html` (open in a browser)
- XML report: `coverage-reports/coverage.xml` (used by SonarQube)

## Continuous Integration

This setup can be integrated into CI/CD pipelines by:

1. Installing the required dependencies
2. Running the test suite with coverage
3. Uploading results to SonarQube
4. Making the build fail if quality gates aren't passed

Example GitHub Actions workflow:

```yaml
name: SonarQube Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  sonarqube:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests with coverage
      run: |
        python -m pytest tests app/tests \
          --cov=app \
          --cov-report=xml:coverage-reports/coverage.xml \
          --junitxml=junit-reports/junit.xml
    
    - name: SonarQube Scan
      uses: SonarSource/sonarqube-scan-action@master
      env:
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
        SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
```

## Troubleshooting

1. **Missing test results**: Ensure tests are running and generating coverage XML reports
2. **Connection issues**: Verify SonarQube server is accessible and token has proper permissions
3. **Invalid configuration**: Check sonar-project.properties file for correct paths and settings
4. **Python version issues**: Make sure your Python version is compatible (3.9+)

## Additional Resources

- [SonarQube Documentation](https://docs.sonarqube.org/latest/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/en/latest/)
- [sonar-scanner Documentation](https://docs.sonarqube.org/latest/analysis/scan/sonarscanner/)