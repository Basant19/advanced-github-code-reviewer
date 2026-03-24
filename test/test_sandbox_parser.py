import pytest
from app.mcp.sandbox_client import SandboxClient

@pytest.fixture
def client():
    return SandboxClient()

def test_parse_standard_git_diff(client):
    """Verifies standard 'a/ b/' prefixes are handled."""
    diff = """diff --git a/app/math.py b/app/math.py
index 12345..67890 100644
--- a/app/math.py
+++ b/app/math.py
@@ -1,2 +1,2 @@
-def add(x,y):
+def add(x, y):
     return x + y
"""
    files = client._parse_diff(diff)
    assert "app/math.py" in files
    assert "def add(x, y):" in files["app/math.py"]

def test_parse_windows_style_diff(client):
    """Verifies 'w/' prefixes (common in some IDEs/Windows) are handled."""
    diff = """--- w/app/core.py
+++ w/app/core.py
@@ -1,1 +1,1 @@
+print("Hello World")
"""
    files = client._parse_diff(diff)
    assert "app/core.py" in files
    assert 'print("Hello World")' in files["app/core.py"]

def test_filter_non_python_files(client):
    """Ensures README.md or other files are skipped as per your logic."""
    diff = """--- a/README.md
+++ b/README.md
+New documentation
--- a/app/main.py
+++ b/app/main.py
+import os
"""
    files = client._parse_diff(diff)
    assert "README.md" not in files
    assert "app/main.py" in files

def test_empty_or_garbage_diff(client):
    """Verifies the 'graceful' return of an empty dict for bad input."""
    assert client._parse_diff("") == {}
    assert client._parse_diff("Just some random text") == {}

def test_multiple_files_in_one_diff(client):
    """Verifies state reset between files."""
    diff = """+++ a/file1.py
+content1
+++ b/file2.py
+content2
"""
    files = client._parse_diff(diff)
    assert files["file1.py"].strip() == "content1"
    assert files["file2.py"].strip() == "content2"