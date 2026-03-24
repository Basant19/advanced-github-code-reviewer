"""
test_parse_diff.py
==================

Purpose:
--------
Validate _parse_diff() against ALL real-world diff formats.

Covers:
✔ diff --git + +++ format (standard GitHub)
✔ Only diff --git (missing +++)
✔ Only +++ header
✔ Non-python files
✔ Deleted files (/dev/null)
✔ Windows paths
✔ Multiple files
✔ Mixed valid + invalid files
✔ Empty diff (error case)

Run:
----
python test_parse_diff.py
"""

import sys
from typing import Optional, Dict
import re


# ------------------------------------------------------------------
# Minimal ParseError (same as your project)
# ------------------------------------------------------------------
class ParseError(Exception):
    pass


# ------------------------------------------------------------------
# Copy your function here (standalone for testing)
# ------------------------------------------------------------------
def parse_diff(diff: str) -> Dict[str, str]:
    if not diff or not diff.strip():
        raise ParseError("Diff is empty — nothing to parse.")

    files: Dict[str, str] = {}
    current_file: Optional[str] = None
    current_lines: list[str] = []

    diff_git_re = re.compile(r"^diff --git a[\\/](.+?) b[\\/](.+)")
    plus_plus_re = re.compile(r"^\+\+\+ (?:b/)?(.+)")

    for line in diff.splitlines():
        new_path = None

        git_match = diff_git_re.match(line)
        plus_match = plus_plus_re.match(line)

        if git_match:
            new_path = git_match.group(2).replace("\\", "/")
        elif plus_match:
            new_path = plus_match.group(1).replace("\\", "/")

        if new_path:
            if new_path != current_file:
                if current_file and current_lines:
                    files[current_file] = "\n".join(current_lines)

                if new_path.endswith(".py") and new_path != "/dev/null":
                    current_file = new_path
                    current_lines = []
                else:
                    current_file = None
                    current_lines = []
            continue

        if current_file is None:
            continue

        if (
            line.startswith("--- ")
            or line.startswith("index ")
            or line.startswith("@@ ")
            or line.startswith("similarity ")
            or line.startswith("rename ")
            or line.startswith("\\ No newline")
        ):
            continue

        if line.startswith("+") and not line.startswith("+++"):
            current_lines.append(line[1:])
        elif line.startswith(" "):
            current_lines.append(line[1:])

    if current_file and current_lines:
        files[current_file] = "\n".join(current_lines)

    if not files:
        raise ParseError("No Python files extracted.")

    return files


# ------------------------------------------------------------------
# TEST CASES
# ------------------------------------------------------------------

def test_standard_diff():
    print("\n[TEST] Standard GitHub diff")

    diff = """diff --git a/app/test.py b/app/test.py
index 123..456 100644
--- a/app/test.py
+++ b/app/test.py
@@ -1,2 +1,2 @@
-def add(a,b):
+def add(a, b):
     return a + b
"""
    result = parse_diff(diff)
    print(result)


def test_only_diff_git():
    print("\n[TEST] Only diff --git (no +++)")

    diff = """diff --git a/main.py b/main.py
@@ -1 +1 @@
-print("hello")
+print("hello world")
"""
    result = parse_diff(diff)
    print(result)


def test_only_plus_header():
    print("\n[TEST] Only +++ header")

    diff = """+++ b/script.py
@@ -0,0 +1,2 @@
+print("hi")
+print("bye")
"""
    result = parse_diff(diff)
    print(result)


def test_non_python_file():
    print("\n[TEST] Non-Python file")

    diff = """diff --git a/readme.md b/readme.md
@@ -1 +1 @@
-Hello
+Hello World
"""
    try:
        parse_diff(diff)
    except ParseError as e:
        print("Expected Error:", e)


def test_deleted_file():
    print("\n[TEST] Deleted file")

    diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ /dev/null
@@ -1 +0,0 @@
-print("deleted")
"""
    try:
        parse_diff(diff)
    except ParseError as e:
        print("Expected Error:", e)


def test_windows_path():
    print("\n[TEST] Windows path")

    diff = r"""diff --git a\app\win.py b\app\win.py
@@ -1 +1 @@
-print("x")
+print("y")
"""
    result = parse_diff(diff)
    print(result)


def test_multiple_files():
    print("\n[TEST] Multiple files")

    diff = """diff --git a/a.py b/a.py
@@ -1 +1 @@
-print("a")
+print("A")

diff --git a/b.py b/b.py
@@ -1 +1 @@
-print("b")
+print("B")
"""
    result = parse_diff(diff)
    print(result)


def test_mixed_files():
    print("\n[TEST] Mixed Python + Non-Python")

    diff = """diff --git a/a.py b/a.py
@@ -1 +1 @@
-print("a")
+print("A")

diff --git a/readme.md b/readme.md
@@ -1 +1 @@
-old
+new
"""
    result = parse_diff(diff)
    print(result)


def test_empty_diff():
    print("\n[TEST] Empty diff")

    try:
        parse_diff("")
    except ParseError as e:
        print("Expected Error:", e)


# ------------------------------------------------------------------
# RUN ALL TESTS
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_standard_diff()
    test_only_diff_git()
    test_only_plus_header()
    test_non_python_file()
    test_deleted_file()
    test_windows_path()
    test_multiple_files()
    test_mixed_files()
    test_empty_diff()

    print("\n✅ ALL TESTS COMPLETED")