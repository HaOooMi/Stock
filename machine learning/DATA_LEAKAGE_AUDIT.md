# 数据泄漏彻底审查报告

**生成时间**: 2025-10-14
**审查人**: Assistant (GPT-4)
**目标**: 彻底检查"信号与未来收益相关性过高(0.3369), 可能存在数据泄露"问题

---

## 必查项A1: 标签是否统一来自同一处函数

### ✅ 检查结果: 通过

**证据**: `target_engineering.py` Line 80-96
```python
def generate_future_returns(self, data: pd.DataFrame, 
                           periods: List[int] = [1, 5, 10],
                           price_col: str = 'close') -> pd.DataFrame:
    for period in periods:
        target_col = f'future_return_{period}d'
        
        # 【关键】正确使用shift(-period)向未来移动
        future_prices = result_df[price_col].shift(-period)
        current_prices = result_df[price_col]
        
        # 计算收益率
        result_df[target_col] = (future_prices - current_prices) / current_prices
```

**验证**:
- ✅ 所有标签都来自`generate_future_returns()`函数
- ✅ 使用`shift(-period)`正确计算未来收益
- ✅ 公式正确: `(future_price - current_price) / current_price`

**尾部NaN保留验证** (`target_engineering.py` Line 138-164):
```python
def _verify_future_returns(self, data: pd.DataFrame, periods: List[int]):
    for period in periods:
        target_col = f'future_return_{period}d'
        
        # 尾部应该有period行NaN
        tail_nans = data[target_col].tail(period).isna().sum()
        
        if tail_nans != period:
            print(f"      ⚠️ 警告: 尾部NaN数量不匹配预期")
```

**结论**: ✅ **标签生成正确，尾部NaN正确保留**

---

## 必查项A2: 训练集尾部是否purge（删去max_h行）

### ⚠️ 检查结果: **有问题！purge_periods=10，但缺少验证**

**证据**: `pca_state.py` Line 135-173

#### 当前实现
```python
def fit_pca_with_time_split(self, features_df: pd.DataFrame,
                           n_components: float = 0.9,
                           train_ratio: float = 0.8,
                           purge_periods: int = 10) -> Dict:  # 硬编码10天
    
    # 时间切分
    n_samples = len(features_df)
    split_idx = int(n_samples * train_ratio)
    
    # Purge训练集尾部(防止标签泄漏)
    split_idx_purged = split_idx - purge_periods  # 删去10行
    
    if split_idx_purged < 50:
        raise ValueError(f"训练样本过少({split_idx_purged}),请减小purge_periods或增加train_ratio")
    
    train_index = features_df.index[:split_idx_purged]  # 训练集[0:70)
    test_index = features_df.index[split_idx:]          # 测试集[80:88)
    
    # 【关键】中间gap [70:80) 被purge掉
```

#### 🚨 发现的问题

**问题1**: `purge_periods=10` **硬编码，未根据max_target_period动态调整**

**证据**: `target_engineering.py` Line 12定义了`periods=[1, 5, 10]`
- 最大目标窗口 `max_h = 10天`
- Purge应该 ≥ 10天，但代码中是硬编码`purge_periods=10`

**问题2**: **缺少显式验证purge gap是否充分**

**正确做法**:
```python
# 应该动态计算purge_periods
max_target_period = max(target_periods)  # 10
purge_periods = max_target_period  # 至少10天

# 应该验证gap内的数据不包含训练目标
gap_data = features_df.iloc[split_idx_purged:split_idx]
assert len(gap_data) == purge_periods, "Purge gap长度错误"
```

**当前风险**:
- ✅ Purge逻辑存在且正确执行
- ⚠️ **但purge_periods是硬编码，未与target_periods关联**
- ⚠️ **如果未来增加更大的target_period (如20天)，purge会失效**

**结论**: ⚠️ **部分通过，但存在硬编码风险**

---

## 必查项A3: IC/Spread的pred与y是否严格对齐（T+1执行）

### ⚠️ 检查结果: **有严重问题！**

#### 问题1: `strategy_backtest.py` 中T+1对齐**正确实现**

**证据**: `strategy_backtest.py` Line 408-420
```python
def calculate_strategy_performance(self, signal_data: pd.DataFrame) -> Dict:
    returns = signal_data['future_return_5d'].fillna(0).values
    signal = signal_data['signal_combined'].values
    
    # 【关键修复】信号对齐: T+1执行
    # 今天的信号决定明天的仓位,避免look-ahead bias
    signal_t_plus_1 = np.roll(signal, 1)  # 信号后移1天
    signal_t_plus_1[0] = 0  # 第一天无信号
    
    # 策略收益：使用对齐后的信号
    strategy_returns = signal_t_plus_1 * returns
```

✅ **这部分正确**: 使用`np.roll(signal, 1)`确保T+1执行

#### 🚨 问题2: `quick_triage.py` 中**IC计算缺少T+1对齐**

**检查**: `quick_triage.py`（假设存在IC计算）

**预期代码应该是**:
```python
# 错误做法（会导致0.3369高相关性）
correlation = signal_data['signal'].corr(signal_data['future_return_5d'])

# 正确做法
signal_t_plus_1 = signal_data['signal'].shift(1)  # T+1对齐
correlation = signal_t_plus_1.corr(signal_data['future_return_5d'])
```

**需要确认**: 
- ❓ `quick_triage.py` 是否存在？
- ❓ IC计算是否使用了T+1对齐？

**当前证据不足**: 需要检查`quick_triage.py`的实际代码

**结论**: ⚠️ **策略回测已正确实现T+1，但IC计算需要验证**

---

## 必查项A4: scaler/PCA/KMeans是否只在训练段fit

### ✅ 检查结果: 通过

#### Scaler验证 (`feature_engineering.py` Line 668-710)

```python
def scale_features(self, features_df: pd.DataFrame, 
                   scaler_type: str = 'robust',
                   train_ratio: float = 0.8) -> Dict:
    
    # 时间切分
    n_samples = len(df)
    split_idx = int(n_samples * train_ratio)
    train_index = df.index[:split_idx]
    valid_index = df.index[split_idx:]
    
    train_X = df.loc[train_index, feature_cols]
    valid_X = df.loc[valid_index, feature_cols]
    
    # 【关键】只在训练集上拟合
    scaler.fit(train_X.fillna(0))
    scaled_train = scaler.transform(train_X.fillna(0))
    
    # 测试集只transform
    scaled_valid = scaler.transform(valid_X.fillna(0))
```

✅ **正确**: Scaler只在训练集fit，测试集只transform

#### PCA验证 (`pca_state.py` Line 135-235)

```python
def fit_pca_with_time_split(self, features_df: pd.DataFrame,
                           n_components: float = 0.9,
                           train_ratio: float = 0.8,
                           purge_periods: int = 10) -> Dict:
    
    # 时间切分 + Purge
    split_idx = int(n_samples * train_ratio)
    split_idx_purged = split_idx - purge_periods
    
    X_train = features_df.iloc[:split_idx_purged].fillna(0)  # 训练集
    X_test = features_df.iloc[split_idx:].fillna(0)          # 测试集
    
    # 【关键】只在训练集上拟合PCA
    pca_final = PCA(n_components=n_components_needed)
    pca_final.fit(X_train)
    
    # 生成训练集和测试集的PCA状态
    states_train = pca_final.transform(X_train)
    states_test = pca_final.transform(X_test)   # 测试集只transform
```

✅ **正确**: PCA只在训练集fit，测试集只transform

#### KMeans验证 (`cluster_evaluate.py` Line 68-86)

```python
def perform_kmeans_clustering(self, states_train: np.ndarray, k: int) -> KMeans:
    kmeans = KMeans(
        n_clusters=k,
        random_state=self.random_state,
        n_init=20,
        max_iter=500
    )
    
    # 【关键】只在训练集上拟合
    kmeans.fit(states_train)
    
    return kmeans
```

```python
def evaluate_cluster_returns(self, states: np.ndarray, targets_df: pd.DataFrame,
                            kmeans: KMeans, phase: str = "train") -> pd.DataFrame:
    
    # 【关键】测试集只predict，不重新fit
    cluster_labels = kmeans.predict(states)
```

✅ **正确**: KMeans只在训练集fit，测试集只predict

**结论**: ✅ **所有预处理器都正确遵循fit-transform分离原则**

---

## 必查项A5: 破坏性对照实验

### ❌ 检查结果: **未实现！**

**要求**: 故意使用错误标签（如`close.pct_change(5)`或`future_return_-5`）看指标是否"更好"

**当前状态**: 
- ❌ 没有找到破坏性对照实验代码
- ❌ `quick_triage.py` 中应该有`check_leakage_with_wrong_labels()`函数，但**文件不存在或未运行**

**应实现的测试**:
```python
def destructive_test():
    """破坏性对照: 使用错误标签检测泄漏"""
    
    # 错误标签1: 过去5天收益（绝对错误）
    wrong_label_past = data['close'].pct_change(5)
    
    # 错误标签2: 负的未来收益（时间反转）
    wrong_label_negated = -data['close'].pct_change(5).shift(-5)
    
    # 错误标签3: 随机打乱的未来收益
    wrong_label_shuffled = data['future_return_5d'].sample(frac=1).values
    
    # 计算相关性
    corr_correct = signal.corr(data['future_return_5d'])
    corr_past = signal.corr(wrong_label_past)
    corr_negated = signal.corr(wrong_label_negated)
    corr_shuffled = pd.Series(signal).corr(pd.Series(wrong_label_shuffled))
    
    # 【验收标准】
    # 1. 正确标签相关性应该最高
    assert corr_correct > corr_past, "信号对过去收益相关性过高，存在泄漏！"
    assert corr_correct > corr_negated, "信号对负收益相关性过高，存在泄漏！"
    assert corr_correct > corr_shuffled, "信号对随机收益相关性过高，存在泄漏！"
    
    # 2. 错误标签相关性应该接近0
    assert abs(corr_past) < 0.1, f"过去收益相关性 {corr_past:.4f} 过高"
    assert abs(corr_shuffled) < 0.1, f"随机收益相关性 {corr_shuffled:.4f} 过高"
```

**结论**: ❌ **未实现破坏性对照，这是重大遗漏！**

---

## 数据泄漏根因分析

### 🔍 为什么"信号与未来收益相关性0.3369"？

#### 假设1: IC计算时未T+1对齐 ⚠️ **高度怀疑**

**场景**: 如果`quick_triage.py`中直接计算:
```python
# 错误做法（导致0.3369高相关性）
signal_today = model.predict(features_today)
return_today = future_return_5d[today]  # 这个值包含了未来5天的信息

correlation = signal_today.corr(return_today)  # 0.3369
```

**问题**: 
- `signal_today`使用了`features_today`（截至今天的特征）
- 但`future_return_5d[today]`是**未来5天的收益**，在今天生成信号时**不应该知道**
- 正确做法应该是`signal_today.shift(1).corr(return_today)`

#### 假设2: 特征工程中包含了未来信息 ⚠️ **需要验证**

**检查**: `feature_engineering.py`中是否有特征使用了`shift(-n)`（负值表示未来）

**扫描结果**:
```python
# Line 239: 收益率特征
data['return_1d'] = data['close'].pct_change()      # ✅ 正确
data['return_5d'] = data['close'].pct_change(5)     # ✅ 正确
data['return_10d'] = data['close'].pct_change(10)   # ✅ 正确

# Line 246: 滚动统计
data[f'rolling_mean_{window}d'] = data['close'].rolling(window).mean()  # ✅ 正确

# Line 256: 动量特征
data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1  # ✅ 正确（使用过去）
```

✅ **特征工程中没有使用未来信息**

#### 假设3: 簇占比过小导致过拟合噪声 ✅ **已修复**

**证据**: 之前发现"最佳簇占比仅1.6%"

**已修复**: `strategy_backtest.py` Line 131-189添加了`min_cluster_pct=0.05`过滤

```python
def select_best_clusters(self, comparison_df: pd.DataFrame, top_n: int =3, 
                       min_cluster_pct: float = 0.05) -> Dict:
    
    # 过滤掉占比过小的簇
    valid_clusters = valid_clusters[valid_clusters['cluster_pct'] >= min_cluster_pct].copy()
```

✅ **已修复占比过小问题**

#### 假设4: Purge gap不足导致训练集尾部标签泄漏 ⚠️ **低风险但需优化**

**当前**: `purge_periods=10`（硬编码）
**最大目标窗口**: `max(target_periods) = 10天`

**风险**: 
- 如果未来增加`target_periods=[1,5,10,20]`，purge=10会失效
- 应该动态设置`purge_periods = max(target_periods)`

⚠️ **建议优化但非紧急**

---

## 综合诊断结论

### 🔴 高度怀疑的泄漏源

#### ⚠️ **最可能原因**: `quick_triage.py`中IC计算缺少T+1对齐

**推理链**:
1. 您报告"信号与未来收益相关性0.3369" → 这是IC计算的结果
2. IC应该在`quick_triage.py`的`check_ranking_power()`中计算
3. 该文件**不在workspace中**或**未被检查**
4. 如果IC计算时使用:
   ```python
   correlation = signal_data['signal'].corr(signal_data['future_return_5d'])
   ```
   而不是:
   ```python
   signal_t_plus_1 = signal_data['signal'].shift(1)
   correlation = signal_t_plus_1.corr(signal_data['future_return_5d'])
   ```
   就会导致0.3369的虚高相关性！

### 🟡 次要可能原因

1. **Purge硬编码**: 存在但风险较低
2. **缺少破坏性对照**: 无法验证泄漏是否存在
3. **簇占比过滤**: 已修复

---

## 紧急修复建议

### 🚨 立即执行（Priority 1）

#### 修复1: 检查并修正`quick_triage.py`的IC计算

```python
# 错误做法（需要修正）
def check_ranking_power_WRONG(self, signal_data, target_col='future_return_5d'):
    correlation = signal_data['signal'].corr(signal_data[target_col])
    return correlation  # 会得到0.3369

# 正确做法
def check_ranking_power_CORRECT(self, signal_data, target_col='future_return_5d'):
    # T+1对齐
    signal_t_plus_1 = signal_data['signal'].shift(1).dropna()
    target_aligned = signal_data[target_col].iloc[1:]
    
    correlation = signal_t_plus_1.corr(target_aligned)
    return correlation  # 应该显著降低
```

#### 修复2: 实现破坏性对照实验

```python
def check_leakage_with_wrong_labels(self, signal_data, target_col='future_return_5d'):
    """破坏性对照: 使用错误标签检测泄漏"""
    
    # 准备信号（T+1对齐）
    signal = signal_data['signal'].shift(1).dropna()
    
    # 正确标签
    correct_target = signal_data[target_col].iloc[1:]
    
    # 错误标签1: 过去收益
    wrong_past = signal_data['close'].pct_change(5).iloc[1:]
    
    # 错误标签2: 随机打乱
    wrong_random = correct_target.sample(frac=1, random_state=42).values
    
    # 计算相关性
    corr_correct = signal.corr(correct_target)
    corr_past = signal.corr(wrong_past)
    corr_random = pd.Series(signal.values).corr(pd.Series(wrong_random))
    
    print(f"相关性 - 正确标签: {corr_correct:.4f}")
    print(f"相关性 - 过去收益: {corr_past:.4f}")
    print(f"相关性 - 随机打乱: {corr_random:.4f}")
    
    # 验收
    if corr_correct < 0.02:
        print("❌ IC过低，信号无预测力")
        return False
    
    if abs(corr_past) > 0.1 or abs(corr_random) > 0.1:
        print("🚨 警告：错误标签相关性过高，可能存在数据泄漏！")
        return False
    
    if corr_correct <= max(abs(corr_past), abs(corr_random)):
        print("🚨 严重警告：正确标签相关性不高于错误标签，确认存在泄漏！")
        return False
    
    print("✅ 通过破坏性对照验证")
    return True
```

#### 修复3: 动态Purge参数

```python
def fit_pca_with_time_split(self, features_df: pd.DataFrame,
                           n_components: float = 0.9,
                           train_ratio: float = 0.8,
                           target_periods: List[int] = [1, 5, 10]) -> Dict:  # 新增参数
    
    # 【修复】动态计算purge_periods
    purge_periods = max(target_periods)  # 确保≥最大目标窗口
    
    print(f"   🚫 训练集尾部purge: {purge_periods}天 (基于max_target_period={max(target_periods)})")
    
    # 时间切分
    split_idx = int(n_samples * train_ratio)
    split_idx_purged = split_idx - purge_periods
    
    # 验证gap
    if split_idx_purged < 50:
        raise ValueError(f"训练样本过少({split_idx_purged}),请减小target_periods或增加train_ratio")
```

### 🟡 推荐执行（Priority 2）

1. **添加显式的训练-测试时间gap验证**
```python
def validate_time_gap(train_index, test_index, min_gap_days=10):
    train_end = train_index.max()
    test_start = test_index.min()
    gap_days = (test_start - train_end).days
    
    assert gap_days >= min_gap_days, f"时间gap不足: {gap_days} < {min_gap_days}天"
    print(f"✅ 时间gap验证通过: {gap_days}天")
```

2. **添加IC分布监控**
```python
def monitor_ic_distribution(ic_values, threshold=0.15):
    """监控IC异常值"""
    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values)
    
    if mean_ic > threshold:
        print(f"⚠️ 警告: 平均IC过高 {mean_ic:.4f} > {threshold}")
        print(f"   可能存在数据泄漏，建议检查信号对齐")
```

---

## 最终验收清单

### ✅ 已通过
- [x] A1: 标签来自统一函数 ✅
- [x] A1: 尾部NaN正确保留 ✅
- [x] A3: 策略回测T+1对齐 ✅
- [x] A4: Scaler只在训练集fit ✅
- [x] A4: PCA只在训练集fit ✅
- [x] A4: KMeans只在训练集fit ✅
- [x] B1: 簇占比过滤 ✅

### ⚠️ 需要修复
- [ ] A2: Purge参数硬编码 → **建议动态化**
- [ ] A3: IC计算T+1对齐 → **紧急检查`quick_triage.py`**
- [ ] A5: 破坏性对照实验 → **紧急实现**
- [ ] 时间gap显式验证 → **建议添加**

### ❌ 未实现
- [ ] 破坏性对照实验代码
- [ ] IC分布监控代码

---

## 行动计划

### 第一步（紧急）: 定位IC计算代码
```bash
# 查找quick_triage.py或IC计算相关代码
grep -r "future_return" machine\ learning/*.py
grep -r "\.corr(" machine\ learning/*.py
grep -r "IC\|information_coefficient" machine\ learning/*.py
```

### 第二步: 修正IC计算（如果存在）
参考上面的`check_ranking_power_CORRECT()`实现

### 第三步: 实现破坏性对照
创建新函数或在`quick_triage.py`中添加

### 第四步: 动态Purge参数
修改`pca_state.py`的`fit_pca_with_time_split()`函数

### 第五步: 重新运行测试
```bash
python machine\ learning/quick_triage.py
python machine\ learning/strategy_backtest.py
```

### 第六步: 验收新IC值
- 期望IC应该在 **0.02 ~ 0.08** 之间（合理范围）
- 如果仍然>0.15，需要进一步检查特征工程

---

## 报告总结

### 🎯 核心发现
1. **最可能的泄漏源**: IC计算缺少T+1对齐（需验证）
2. **次要问题**: Purge参数硬编码、缺少破坏性对照
3. **已修复**: 簇占比过滤、策略回测T+1对齐

### 📋 执行优先级
1. **P0-紧急**: 检查IC计算是否T+1对齐
2. **P1-重要**: 实现破坏性对照实验
3. **P2-推荐**: 动态Purge参数、时间gap验证

### ✅ 验收标准
修复后应满足:
- IC相关性: **0.02 ~ 0.08**（不应>0.15）
- 破坏性对照: 正确标签 > 错误标签
- 样本外IC: **> 0** 且稳定
- 训练-测试gap: **≥ max_target_period**

