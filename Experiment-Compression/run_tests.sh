#!/bin/bash
# Quick test runner for compression_eval_iterative tests
# Usage: ./run_tests.sh [pytest_args]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest is not installed. Install with: pip install pytest"
    exit 1
fi

echo "🧪 Running compression_eval_iterative tests..."
echo "=================================================="
echo ""

# Run pytest with provided arguments or defaults
if [ $# -eq 0 ]; then
    # Default: verbose mode, show slowest tests
    pytest tests/ -v --tb=short -x
else
    # Pass through all arguments
    pytest tests/ "$@"
fi

exit_code=$?

echo ""
echo "=================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Tests failed (exit code: $exit_code)"
fi

exit $exit_code
