import sys
from pathlib import Path

# 项目根本地化（无 pyproject/editable install，测试需手动加路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
