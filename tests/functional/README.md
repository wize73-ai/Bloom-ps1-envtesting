# Functional Tests for CasaLingua

This directory contains functional tests for the CasaLingua application. Functional tests verify the overall behavior of the application by simulating real user workflows.

## Overview

Functional tests verify that the system works from the user's perspective. They test the system as a whole, including the API, database, models, and other components.

## Test Categories

- **API Tests**: Test the API endpoints directly
- **Workflow Tests**: Test complete user workflows
- **End-to-End Tests**: Test the system from start to finish

## Running the Tests

To run all functional tests:

```bash
python -m pytest tests/functional/
```

To run a specific category of tests:

```bash
python -m pytest tests/functional/api/
python -m pytest tests/functional/workflows/
```

## Test Requirements

- The CasaLingua server must be running
- Required models should be downloaded
- Database should be initialized

## Test Data

The `test_data` directory contains realistic test data for the functional tests, including:

- Sample text files in different languages
- Sample documents in different formats
- Expected translation outputs