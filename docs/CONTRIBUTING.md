# Contributing to Linucast

Thank you for considering contributing to Linucast! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your forked repository to your local machine
3. Create a new branch for your feature or bug fix

## Development Process

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/Adelkazzaz/Linucast.git
cd linucast

# Set up Python environment
cd python_core
poetry install --with dev

# Build C++ components
cd ../cpp_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Code Style and Standards

- **Python**: Follow PEP 8 style guide
- **C++**: Follow the Google C++ Style Guide
- Use descriptive variable names and comments
- Write clear docstrings for functions and classes

## Pull Request Process

1. Ensure your code adheres to the project's style guidelines
2. Update documentation (README.md, docstrings, etc.) with details of changes
3. Add tests that verify your changes
4. Make sure all tests pass before submitting your PR
5. Include a clear description of the changes and their purpose
6. Link any related issues in the pull request description

## Testing

- Add tests for new features and bug fixes
- Run the test suite before submitting a PR:

```bash
cd python_core
poetry run pytest
```

## Documentation

- Update documentation when changing functionality
- Use clear and concise language
- Include examples where appropriate
- Document public APIs, classes, and functions

## Issue Reporting

When reporting issues, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Screenshots or log output if applicable

## Feature Requests

For feature requests, please include:

- A clear description of the feature
- Why this feature would be useful to many users
- Any implementation ideas you might have

## Code Review Process

1. At least one project maintainer will review your code
2. Address any comments or suggestions from reviewers
3. Once approved, a maintainer will merge your PR

## License

By contributing to Linucast, you agree that your contributions will be licensed under the Apache License 2.0.
