from setuptools import setup, find_packages

setup(
    name="my_strategies",           # 你的包名（随便取，但最好唯一）
    version="0.0.1",                # 版本号
    packages=find_packages(),       # 自动找到含有 __init__.py 的文件夹
    install_requires=[
        "vnpy",                     # 依赖 vn.py
    ],
)