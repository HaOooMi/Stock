# 核心依赖说明

## 文件说明

项目提供 4 个环境配置文件，根据需求选择：

### 推荐使用（最小化）

1. **environment_minimal.yml** - Conda 最小化配置
   - 只包含项目运行的核心依赖（约 15 个包）
   - 版本要求宽松，兼容性好
   - 适合新用户和部署环境

2. **requirements_minimal.txt** - pip 最小化配置
   - 与 environment_minimal.yml 对应的 pip 版本
   - 适合不使用 conda 的用户

### 完整配置（开发环境）

3. **environment.yml** - 完整开发环境
   - 与开发者环境完全一致（Python 3.9.23）
   - 包含所有版本精确固定
   - 适合需要完全复现结果的场景

4. **requirements.txt** - pip 完整依赖
   - 导出自实际 ML 环境
   - 版本固定，可能兼容性较差

---

## 依赖分析

基于代码 `import` 语句分析，项目实际使用：

### 核心数据处理
- pandas, numpy, scipy

### 机器学习
- scikit-learn, lightgbm, xgboost

### 数据库
- influxdb-client, PyMySQL

### 可视化
- matplotlib, seaborn

### 技术指标
- TA-Lib（需预装 C 库）

### 工具
- PyYAML, tqdm, joblib

### 数据源
- akshare, tushare

### 其他
- pyarrow（Parquet 格式）
- openpyxl（Excel 支持）

---

## 快速安装

**Conda 用户（推荐）：**

```bash
conda env create -f environment_minimal.yml
conda activate stock_ml
```

**pip 用户：**

```bash
pip install -r requirements_minimal.txt
```

---

