import sys
from pathlib import Path
import glob

if __name__ == "__main__":
    # 获取命令行参数，忽略脚本名
    patterns = sys.argv[1:]
    files = []
    for pattern in patterns:
        # glob 支持通配符，recursive=True 支持 **
        files.extend(glob.glob(pattern, recursive=True))
    # 去重并排序
    unique_files = sorted(set(files))
    # 转为 Path 对象并输出
    for f in unique_files:
        print(Path(f))
