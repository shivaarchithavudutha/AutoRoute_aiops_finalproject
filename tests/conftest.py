import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # for "gateway"
sys.path.insert(0, os.path.join(ROOT, "src"))  # for "text_utils", "prepare", ...
