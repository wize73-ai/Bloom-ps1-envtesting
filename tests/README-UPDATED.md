# CasaLingua Testing Suite

This directory contains tests for the CasaLingua application. The tests are organized into different categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test how components work together
- **Functional Tests**: Test the system as a whole from the user's perspective

## Test Organization

The tests are organized in the following directory structure:

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── models/           # Tests for model-related functionality
│   ├── pipeline/         # Tests for pipeline components
│   ├── rag/              # Tests for RAG components
│   ├── storage/          # Tests for storage components
│   └── utils/            # Tests for utility functions
│
├── integration/          # Integration tests for component interactions
│   ├── api/              # Tests for API integration
│   └── models/           # Tests for model integration
│
├── functional/           # Functional tests for system behavior
│   ├── api/              # Tests for API endpoints
│   ├── pipeline/         # Tests for pipeline functionality
│   ├── server/           # Tests for server startup and initialization
│   ├── workflows/        # Tests for end-to-end workflows
│   └── test_data/        # Test data for functional tests
│
├── scripts/              # Shell scripts and test scripts
│
├── results/              # Contains test result data
│
├── conftest.py           # Common test fixtures and utilities
├── run_tests.py          # Master test runner script
├── migrate_tests.py      # Script used to migrate tests from root directory
└── check_imports.py      # Script to check for import issues
```

## Running Tests

### Using the Master Test Runner

You can use the `run_tests.py` script to run tests in various ways:

```bash
# Run all tests
python tests/run_tests.py

# Run tests by type
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration

# Run tests by category
python tests/run_tests.py --type unit --category models
python tests/run_tests.py --type integration --category api

# Run a specific test file
python tests/run_tests.py --file tests/unit/pipeline/test_language_detection.py

# Set environment
python tests/run_tests.py --env production
```

### Running Unit Tests Directly

You can also use pytest directly to run unit tests:

```bash
# Run all unit tests
python -m pytest tests/unit/

# Run tests for a specific component
python -m pytest tests/unit/utils/
python -m pytest tests/unit/test_main.py
```

### Running Functional Tests

Use the functional test runner script to run functional tests:

```bash
# Run all functional tests
python scripts/run_functional_tests.py

# Run tests for a specific category
python scripts/run_functional_tests.py --category api
python scripts/run_functional_tests.py --category workflows
python scripts/run_functional_tests.py --category pipeline
python scripts/run_functional_tests.py --category server

# Run tests with a specific pattern
python scripts/run_functional_tests.py --pattern translation

# Start a server for testing
python scripts/run_functional_tests.py --server

# Generate JUnit XML reports
python scripts/run_functional_tests.py --junit
```

## Coverage Reports

The tests are configured to generate coverage reports in HTML and XML formats. These reports show which parts of the code are being tested and which are not.

### Generating Coverage Reports

```bash
# Generate coverage reports for unit tests
python -m pytest tests/unit/ --cov=app --cov-report=html --cov-report=xml

# Generate coverage reports for specific modules
python -m pytest tests/unit/ --cov=app.main --cov=app.utils --cov-report=html --cov-report=xml

# Use the core tests runner
python scripts/run_core_tests.py
```

The coverage reports will be generated in the following locations:

- HTML: `coverage-reports/html/index.html`
- XML: `coverage-reports/coverage.xml`

## Writing New Tests

When writing new tests, follow these guidelines:

1. **Test Organization**: Place tests in the appropriate directory based on their type and the component they're testing.
2. **Test Isolation**: Unit tests should be isolated and not depend on other components.
3. **Meaningful Names**: Give tests meaningful names that describe what they're testing.
4. **Setup and Teardown**: Use fixtures for setup and teardown to avoid code duplication.
5. **Assertions**: Use meaningful assertions that clearly indicate what's being tested.
6. **Test Data**: Use realistic test data that covers normal cases, edge cases, and error cases.

## Test Dependencies

Some tests, especially functional tests, require external dependencies to be available:

- **Server**: Some tests require the CasaLingua server to be running
- **Models**: Some tests require models to be downloaded and available
- **Database**: Some tests require a database connection

If these dependencies are not available, the tests will be skipped or will fail with informative error messages.

## Important Notes

1. All tests now use absolute imports and should work when run from the project root directory.

2. The test scripts have been moved to the `tests/scripts` directory but may need to be run from the root directory to work correctly.

3. If you encounter issues with tests not finding modules, make sure you're running them from the project root directory, not from inside the tests directory.