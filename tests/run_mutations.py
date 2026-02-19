#!/usr/bin/env python3
"""Targeted mutation testing for LLM API key authentication.

Applies specific mutations to llm_service.py and verifies our tests catch them.
Each mutation represents a realistic bug that could be introduced.
"""

import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).parent.parent / "calendar_agent" / "llm_service.py"
TEST_CMD = [
    sys.executable, "-m", "pytest",
    "tests/test_llm_service.py", "-x", "-q", "--tb=short", "--no-header",
]

# (description, original_text, mutated_text)
MUTATIONS = [
    (
        "Remove API key storage in __init__ (always use module-level default)",
        'self.api_key = api_key if api_key is not None else LLM_API_KEY',
        'self.api_key = LLM_API_KEY',
    ),
    (
        "Never send Authorization header (delete the if-block body)",
        '            if self.api_key:\n                headers["Authorization"] = f"Bearer {self.api_key}"',
        '            pass  # mutant: auth header removed',
    ),
    (
        "Always send Authorization header (remove the if guard)",
        '            if self.api_key:\n                headers["Authorization"] = f"Bearer {self.api_key}"',
        '            headers["Authorization"] = f"Bearer {self.api_key}"',
    ),
    (
        "Wrong auth scheme (Basic instead of Bearer)",
        'headers["Authorization"] = f"Bearer {self.api_key}"',
        'headers["Authorization"] = f"Basic {self.api_key}"',
    ),
    (
        "Don't pass headers to request",
        '                    headers=headers,',
        '                    # headers=headers,  # mutant: headers dropped',
    ),
    (
        "Remove Content-Type header",
        'headers = {"Content-Type": "application/json"}',
        'headers = {}',
    ),
    (
        "Invert the api_key truthiness check",
        '            if self.api_key:',
        '            if not self.api_key:',
    ),
    (
        "Hardcode empty api_key (ignore constructor arg)",
        'self.api_key = api_key if api_key is not None else LLM_API_KEY',
        'self.api_key = ""',
    ),
    (
        "Use wrong env var name for API key",
        'LLM_API_KEY = os.environ.get("LLM_API_KEY", "")',
        'LLM_API_KEY = os.environ.get("LLM_SECRET_KEY", "")',
    ),
]


def run_tests() -> bool:
    """Run tests, return True if they PASS."""
    result = subprocess.run(TEST_CMD, capture_output=True, text=True, cwd=SOURCE.parent.parent)
    return result.returncode == 0


def main():
    original = SOURCE.read_text()

    # Sanity: tests pass on unmodified source
    print("Baseline: running tests on unmodified source...")
    if not run_tests():
        print("FAIL: Tests don't pass on unmodified source! Fix tests first.")
        sys.exit(1)
    print("  OK — baseline passes\n")

    killed = 0
    survived = 0
    errors = 0

    for i, (desc, old, new) in enumerate(MUTATIONS, 1):
        print(f"Mutation {i}/{len(MUTATIONS)}: {desc}")

        if old not in original:
            print(f"  ERROR — could not find target text in source")
            errors += 1
            continue

        # Apply mutation
        mutated = original.replace(old, new, 1)
        SOURCE.write_text(mutated)

        try:
            if run_tests():
                print(f"  SURVIVED — tests did NOT catch this mutation!")
                survived += 1
            else:
                print(f"  KILLED — tests caught the mutation")
                killed += 1
        finally:
            # Always restore original
            SOURCE.write_text(original)

    print(f"\n{'='*60}")
    print(f"Mutation Testing Results")
    print(f"{'='*60}")
    print(f"  Total mutations: {len(MUTATIONS)}")
    print(f"  Killed:          {killed}")
    print(f"  Survived:        {survived}")
    print(f"  Errors:          {errors}")
    effective = len(MUTATIONS) - errors
    score = (killed / effective * 100) if effective else 0
    print(f"  Mutation score:  {score:.0f}% ({killed}/{effective})")
    print(f"{'='*60}")

    if survived > 0:
        print("\nWARNING: Some mutants survived — tests may have gaps.")
        sys.exit(1)
    elif errors > 0:
        print("\nWARNING: Some mutations couldn't be applied.")
        sys.exit(1)
    else:
        print("\nAll mutants killed — tests are robust.")
        sys.exit(0)


if __name__ == "__main__":
    main()
