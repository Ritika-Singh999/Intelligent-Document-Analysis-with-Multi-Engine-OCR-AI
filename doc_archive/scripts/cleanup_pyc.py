"""Cleanup script to remove .pyc files and __pycache__ directories in the repo."""
import os
import sys
import shutil

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
removed = []
for dirpath, dirnames, filenames in os.walk(root):
    # Remove __pycache__ directories
    if "__pycache__" in dirnames:
        cache_dir = os.path.join(dirpath, "__pycache__")
        try:
            shutil.rmtree(cache_dir)
            removed.append(cache_dir)
        except Exception as e:
            print(f"Failed to remove {cache_dir}: {e}")
    # Remove .pyc files
    for fn in filenames:
        if fn.endswith('.pyc'):
            path = os.path.join(dirpath, fn)
            try:
                os.remove(path)
                removed.append(path)
            except Exception as e:
                print(f"Failed to remove {path}: {e}")

print("Removed items:")
for r in removed:
    print(r)
print(f"Total removed: {len(removed)}")
sys.exit(0)
