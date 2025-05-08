# Unit Tests for CasaLingua

This directory contains unit tests for the CasaLingua application.

## Overview

The unit tests in this directory test individual components of the application in isolation.

## File Structure

- `models/` - Tests for model-related functionality
- `pipeline/` - Tests for the processing pipeline components
- `rag/` - Tests for RAG (Retrieval-Augmented Generation) components
- `storage/` - Tests for storage and persistence-related components
- `utils/` - Tests for utility functions

## Key Tests

### Main Application Entry Point Tests

The `test_main.py` file contains tests for the main application entry point (`app/main.py`). These tests focus on:

1. Testing the existence and structure of the lifespan context manager
2. Testing the application properties
3. Testing the ModelSizeConfig class
4. Testing the GPUInfo class
5. Testing the EnhancedHardwareInfo class

These tests focus on the most testable components of the main module without requiring complex initialization.

## Code Coverage

As of the latest update, we've achieved the following coverage:

- app/main.py: 23% coverage (up from 0%)
- Overall project: 22% coverage (up from 10%)

## Running the Tests

To run the tests, use the following command:

```bash
pytest tests/unit/
```

To run a specific test file:

```bash
pytest tests/unit/test_main.py
```

To run tests with coverage information:

```bash
pytest --cov=app tests/unit/
```

## Test Development Approach

When developing tests for complex modules like `app/main.py`, we followed these principles:

1. Start with the most isolated and self-contained components
2. Use simplified mocking approaches for complex dependencies
3. Test the presence and structure of key functions and classes
4. Gradually expand test coverage as we understand the codebase better

## Notes

Some components are difficult to test in isolation due to complex dependencies. For these components, we've opted for simplified tests that verify their existence and basic structure, rather than attempting to test their full functionality.