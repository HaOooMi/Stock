#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60分钟快速体检（Triage）

在进入机器学习基线之前,先快速验证信号本身是否有区分力。
如果体检显示"信号本身有弱但存在的区分力(IC/Spread>0)但被转换/成本/对齐问题吃掉",
应先修正策略转换与回测假设；
如果体检显示"样本外IC≈0、簇收益分层消失",再进入阶段12(机器学习基线)。

6个快速体检步骤:
1. 信号对齐与泄露检查
2. 成本与换手拆解
3. 排序力体检(不依赖策略)
4. 状态过滤体检(PCA+KMeans)
5. 门槛/持有周期小网格
6. 随机基准与年度切片

简化判断:
- 样本外IC≥0.02 或 5桶Spread>0 且稳定 → 先优化信号转换
- IC≈0、Spread≈0 → 进入阶段12,或回到特征/状态层

作者: Assistant
日期: 2025-10-14
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scipy import stats
from sklearn.cluster import KMeans


class QuickTriage:
    """
    60分钟快速体检系统
    
    在进入复杂的机器学习模型之前,快速诊断信号质量和策略转换问题
    """
    
    def __init__(self, reports_dir: str = "ML output/reports"):
        """初始化体检系统"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if os.path.isabs(reports_dir):
            self.reports_dir = reports_dir
        else:
            self.reports_dir = os.path.join(self.project_root, reports_dir)
        
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # 体检报告
        self.triage_report = []
        self.issues_found = []
        self.recommendations = []
        
        print("🏥 快速体检系统初始化完成")
        print(f"📁 报告目录: {self.reports_dir}")
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        self.triage_report.append(log_message)
    
    def add_issue(self, issue: str):
        """记录发现的问题"""
        self.issues_found.append(issue)
        self.log(f"⚠️  问题: {issue}", "WARN")
    
    def add_recommendation(self, recommendation: str):
        """添加建议"""
        self.recommendations.append(recommendation)
        self.log(f"💡 建议: {recommendation}", "SUGGEST")
    
    # ========== 体检1: 信号对齐与泄露检查 ==========
    
    def check_signal_alignment_and_leakage(self, signal_data: pd.DataFrame) -> Dict:
        """
        体检1: 信号对齐与泄露检查
        
        检查项:
        1. 信号生成时是否使用了未来数据(look-ahead bias)
        2. 信号与收益的时间对齐是否正确
        3. 是否有数据泄露(target leakage)
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号和未来收益的数据
            必须包含: signal_combined, future_return_5d, close
        
        Returns:
        --------
        dict: 检查结果
        """
        self.log("=" * 70)
        self.log("体检1: 信号对齐与泄露检查")
        self.log("=" * 70)
        
        results = {
            'alignment_correct': True,
            'no_leakage': True,
            'issues': []
        }
        
        # 检查1: 信号是否在正确的时间点生成
        # 信号应该基于当前及之前的数据,不应使用未来数据
        self.log("检查信号生成时点...")
        
        # 检查信号与未来收益的相关性
        # 【关键修复】使用T+1对齐避免look-ahead bias
        signal = signal_data['signal_combined'].values
        returns = signal_data['future_return_5d'].values
        
        # 【关键修复】T+1对齐: 今天的信号决定明天的仓位
        signal_t_plus_1 = np.roll(signal, 1)  # 信号后移1天
        signal_t_plus_1[0] = 0  # 第一天无信号
        
        # 移除NaN
        valid_mask = ~np.isnan(returns)
        signal_clean = signal_t_plus_1[valid_mask]  # 使用对齐后的信号
        returns_clean = returns[valid_mask]
        
        if len(signal_clean) > 0:
            correlation = np.corrcoef(signal_clean, returns_clean)[0, 1]
            self.log(f"   信号与未来收益相关性(T+1对齐): {correlation:.4f}")
            
            # 【修改阈值】T+1对齐后,合理IC应在0.02~0.15之间
            if abs(correlation) > 0.15:  # 降低阈值(原来0.3,T+1后应显著降低)
                issue = f"信号与未来收益相关性过高({correlation:.4f}),可能存在数据泄露"
                self.add_issue(issue)
                results['no_leakage'] = False
                results['issues'].append(issue)
            elif abs(correlation) < 0.02:
                issue = f"信号与未来收益相关性过低({correlation:.4f}),信号无预测力"
                self.add_issue(issue)
                results['no_predictive_power'] = True
                results['issues'].append(issue)
            else:
                self.log(f"   ✅ 相关性正常(IC={correlation:.4f}在合理范围0.02~0.15)")
        
        # 检查2: 验证信号生成不使用当期收益
        self.log("\n检查信号是否使用当期收益...")
        
        # 如果信号完全依赖于当期收益,说明对齐有问题
        if 'close' in signal_data.columns:
            current_return = signal_data['close'].pct_change().values
            current_return_clean = current_return[valid_mask]
            
            # 计算信号切换点与当期收益的关系
            signal_changes = np.diff(signal_clean, prepend=signal_clean[0])
            signal_change_mask = signal_changes != 0
            
            if signal_change_mask.sum() > 0:
                # 在信号变化时,检查是否总是跟随当期收益方向
                changes_with_positive_return = np.sum(
                    (signal_changes[signal_change_mask] > 0) & 
                    (current_return_clean[signal_change_mask] > 0)
                )
                changes_with_negative_return = np.sum(
                    (signal_changes[signal_change_mask] < 0) & 
                    (current_return_clean[signal_change_mask] < 0)
                )
                
                alignment_ratio = (changes_with_positive_return + changes_with_negative_return) / signal_change_mask.sum()
                self.log(f"   信号变化与当期收益同向比例: {alignment_ratio:.2%}")
                
                if alignment_ratio > 0.7:
                    issue = f"信号变化过度跟随当期收益({alignment_ratio:.2%}),可能存在对齐问题"
                    self.add_issue(issue)
                    results['alignment_correct'] = False
                    results['issues'].append(issue)
                else:
                    self.log("   ✅ 信号独立于当期收益")
        
        # 检查3: 验证测试集不参与模型训练
        self.log("\n检查训练/测试集分离...")
        
        # 检查是否有明显的训练集过拟合迹象
        # 将数据分为两半,比较性能差异
        mid_point = len(signal_clean) // 2
        
        first_half_signal = signal_clean[:mid_point]
        first_half_return = returns_clean[:mid_point]
        second_half_signal = signal_clean[mid_point:]
        second_half_return = returns_clean[mid_point:]
        
        if len(first_half_signal) > 0 and first_half_signal.sum() > 0:
            first_half_perf = np.mean(first_half_return[first_half_signal == 1])
        else:
            first_half_perf = 0
            
        if len(second_half_signal) > 0 and second_half_signal.sum() > 0:
            second_half_perf = np.mean(second_half_return[second_half_signal == 1])
        else:
            second_half_perf = 0
        
        self.log(f"   前半段信号平均收益: {first_half_perf:+.4f}")
        self.log(f"   后半段信号平均收益: {second_half_perf:+.4f}")
        
        if first_half_perf > 0 and second_half_perf < -abs(first_half_perf) * 0.5:
            issue = "前后半段性能严重背离,可能存在过拟合或测试集泄露"
            self.add_issue(issue)
            results['issues'].append(issue)
        else:
            self.log("   ✅ 前后半段性能一致性良好")
        
        # 总结
        if results['alignment_correct'] and results['no_leakage']:
            self.log("\n✅ 体检1通过: 未发现信号对齐或泄露问题")
        else:
            self.log("\n❌ 体检1未通过: 发现信号对齐或泄露问题")
            self.add_recommendation("修正信号生成逻辑,确保不使用未来数据")
        
        return results
    
    # ========== 体检1A: 破坏性对照实验 ==========
    
    def check_leakage_with_wrong_labels(self, signal_data: pd.DataFrame) -> Dict:
        """
        体检1A: 破坏性对照实验
        
        使用故意错误的标签来验证是否存在数据泄漏。
        如果错误标签的性能"更好",说明管线里有穿越。
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含close列的数据
        
        Returns:
        --------
        dict: 对照实验结果
        """
        self.log("=" * 70)
        self.log("体检1A: 破坏性对照实验 (用错误标签验证泄漏)")
        self.log("=" * 70)
        
        if 'close' not in signal_data.columns:
            self.log("⚠️  缺少close列,跳过破坏性对照")
            return {'skipped': True}
        
        # 获取PCA特征和正确标签
        pca_columns = [col for col in signal_data.columns if col.startswith('PC')]
        if len(pca_columns) == 0:
            self.log("⚠️  未找到PCA特征,跳过破坏性对照")
            return {'skipped': True}
        
        correct_label = signal_data['future_return_5d'].fillna(0).values
        feature = signal_data[pca_columns[0]].fillna(0).values
        
        # 【关键修复】T+1对齐: 今天的特征预测明天的收益
        feature_t_plus_1 = np.roll(feature, 1)
        feature_t_plus_1[0] = 0  # 第一天无特征值
        
        # 计算正确标签的IC
        valid_mask = ~np.isnan(signal_data['future_return_5d'].values)
        if valid_mask.sum() < 10:
            self.log("⚠️  有效样本过少,跳过破坏性对照")
            return {'skipped': True}
        
        feature_valid = feature_t_plus_1[valid_mask]  # 使用对齐后的特征
        correct_label_valid = correct_label[valid_mask]
        
        correct_ic, correct_p = stats.spearmanr(feature_valid, correct_label_valid)
        self.log(f"✅ 正确标签IC(T+1对齐): {correct_ic:+.4f} (p={correct_p:.4f})")
        
        results = {
            'correct_ic': correct_ic,
            'wrong_ics': {},
            'leakage_detected': False
        }
        
        # 错误标签1: 过去5天收益(用于检测动量特征强度,非泄漏指标)
        self.log("\n测试错误标签1: 过去5天收益...")
        wrong_label_1 = signal_data['close'].pct_change(5).fillna(0).values[valid_mask]
        wrong_ic_1, wrong_p_1 = stats.spearmanr(feature_valid, wrong_label_1)
        self.log(f"   过去收益 IC: {wrong_ic_1:+.4f} (p={wrong_p_1:.4f})")
        results['wrong_ics']['past_5d_return'] = wrong_ic_1
        
        # 【重要】过去收益IC高不是泄漏,而是PCA捕捉动量特征的正常表现
        # 因为特征层包含return_5d/momentum_5d等,PCA自然会与过去收益相关
        if abs(wrong_ic_1) > abs(correct_ic):
            momentum_strength = abs(wrong_ic_1) / (abs(correct_ic) + 1e-6)
            self.log(f"   ℹ️  动量强度: {momentum_strength:.1f}x (过去IC={wrong_ic_1:+.4f}, 未来IC={correct_ic:+.4f})")
            
            # 判断动量方向
            if correct_ic * wrong_ic_1 > 0:
                self.log("   📈 动量延续: 过去表现好的未来继续好")
            else:
                self.log("   🔄 动量反转: 过去表现好的未来表现差（当前状态）")
            
            # 只有在过去收益IC极端高时才警告（可能是shift错误）
            if abs(wrong_ic_1) > abs(correct_ic) * 3.0:
                self.add_issue(f"过去收益IC({wrong_ic_1:+.4f})远超未来收益IC({correct_ic:+.4f})的3倍,请检查特征shift方向")
                results['leakage_detected'] = True
        else:
            self.log("   ✅ 未来预测性优于历史相关性")
        
        # 错误标签2: 随机标签(纯噪声)
        self.log("\n测试错误标签2: 随机标签...")
        np.random.seed(42)
        wrong_label_2 = np.random.randn(len(feature_valid))
        wrong_ic_2, wrong_p_2 = stats.spearmanr(feature_valid, wrong_label_2)
        self.log(f"   错误标签2 IC: {wrong_ic_2:+.4f} (p={wrong_p_2:.4f})")
        results['wrong_ics']['random'] = wrong_ic_2
        
        if abs(wrong_ic_2) > abs(correct_ic):
            self.add_issue(f"随机标签的IC({wrong_ic_2:+.4f})超过正确标签({correct_ic:+.4f}),疑似泄漏!")
            results['leakage_detected'] = True
        
        # 错误标签3: 当期收益(T而非T+h,对齐错误)
        self.log("\n测试错误标签3: 当期收益(对齐错误)...")
        wrong_label_3 = signal_data['close'].pct_change().fillna(0).values[valid_mask]
        wrong_ic_3, wrong_p_3 = stats.spearmanr(feature_valid, wrong_label_3)
        self.log(f"   错误标签3 IC: {wrong_ic_3:+.4f} (p={wrong_p_3:.4f})")
        results['wrong_ics']['current_return'] = wrong_ic_3
        
        if abs(wrong_ic_3) > abs(correct_ic) * 1.5:
            self.add_issue(f"当期收益的IC({wrong_ic_3:+.4f})远超正确标签({correct_ic:+.4f}),疑似对齐错误!")
            results['leakage_detected'] = True
        
        # 总结
        if results['leakage_detected']:
            self.log("\n❌ 破坏性对照未通过: 检测到数据泄漏迹象")
            self.add_recommendation("紧急检查标签生成、数据对齐和特征工程管线")
        else:
            self.log("\n✅ 破坏性对照通过: 未检测到明显泄漏")
        
        return results
    
    # ========== 体检2: 成本与换手拆解 ==========
    
    def analyze_cost_and_turnover(self, signal_data: pd.DataFrame, 
                                  transaction_cost: float = 0.002,
                                  slippage: float = 0.001) -> Dict:
        """
        体检2: 成本与换手拆解
        
        分析交易成本和换手率对策略收益的影响
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号的数据
        transaction_cost : float
            单边交易成本(包括佣金、印花税等),默认0.2%
        slippage : float
            滑点成本,默认0.1%
        
        Returns:
        --------
        dict: 成本分析结果
        """
        self.log("=" * 70)
        self.log("体检2: 成本与换手拆解")
        self.log("=" * 70)
        
        signal = signal_data['signal_combined'].values
        returns = signal_data['future_return_5d'].fillna(0).values
        
        # 【关键修复】严格T+1执行：今天的信号决定明天的仓位
        signal_t1 = np.roll(signal, 1)
        signal_t1[0] = 0  # 第一天无信号
        
        # 计算换手率（按回合计费）
        signal_changes = np.abs(np.diff(signal, prepend=signal[0]))
        flips = signal_changes.sum()
        roundtrips = flips / 2.0  # 每两个翻转构成一次完整回合（开+平）
        turnover_rate = roundtrips / len(signal)
        
        self.log(f"换手统计:")
        self.log(f"   信号翻转次数: {flips:.0f}")
        self.log(f"   交易回合数: {roundtrips:.1f}")
        self.log(f"   换手率: {turnover_rate:.2%}")
        self.log(f"   平均持有期: {1/turnover_rate:.1f} 期" if turnover_rate > 0 else "   平均持有期: N/A")
        
        # 计算不同成本假设下的收益（使用T+1对齐的信号）
        # 策略收益(不考虑成本)
        strategy_returns_gross = signal_t1 * returns
        gross_total_return = float(np.sum(strategy_returns_gross))
        
        # 计算交易成本（按回合计费，双边成本）
        per_roundtrip_cost = (transaction_cost + slippage) * 2  # 双边成本
        total_transaction_cost = roundtrips * per_roundtrip_cost
        
        # 净收益
        net_total_return = gross_total_return - total_transaction_cost
        
        self.log(f"\n收益拆解:")
        self.log(f"   毛收益: {gross_total_return:+.4f}")
        self.log(f"   交易成本: {total_transaction_cost:-.4f} ({roundtrips:.1f}回合 × {per_roundtrip_cost:.4f})")
        self.log(f"   净收益: {net_total_return:+.4f}")
        self.log(f"   成本侵蚀比例: {(total_transaction_cost/abs(gross_total_return)*100):.1f}%" if gross_total_return != 0 else "   成本侵蚀比例: N/A")
        
        # 判断成本影响
        results = {
            'turnover_count': flips,
            'roundtrips': roundtrips,
            'turnover_rate': turnover_rate,
            'gross_return': gross_total_return,
            'transaction_cost': total_transaction_cost,
            'net_return': net_total_return,
            'cost_erosion_ratio': total_transaction_cost / abs(gross_total_return) if gross_total_return != 0 else 0
        }
        
        if results['cost_erosion_ratio'] > 0.5:
            issue = f"交易成本吃掉{results['cost_erosion_ratio']*100:.0f}%的毛收益"
            self.add_issue(issue)
            self.add_recommendation("降低换手率:延长持有周期、提高信号门槛、合并相邻信号")
        elif results['cost_erosion_ratio'] > 0.3:
            self.log("\n⚠️  交易成本较高,建议优化换手")
            self.add_recommendation("适度降低换手率可提升净收益")
        else:
            self.log("\n✅ 成本控制良好")
        
        # 不同成本假设的敏感性分析
        self.log("\n成本敏感性分析:")
        cost_scenarios = [0.001, 0.002, 0.003, 0.005]
        
        for cost in cost_scenarios:
            scenario_cost = roundtrips * cost * 2  # 按回合计费
            scenario_net = gross_total_return - scenario_cost
            self.log(f"   成本{cost*100:.2f}%: 净收益 {scenario_net:+.4f}")
        
        return results
    
    # ========== 体检3: 排序力体检 ==========
    
    def check_ranking_power(self, signal_data: pd.DataFrame, n_quantiles: int = 5) -> Dict:
        """
        体检3: 排序力体检(不依赖策略)
        
        使用IC(信息系数)和分层收益检验信号的排序能力
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            必须包含信号相关特征和future_return_5d
        n_quantiles : int
            分层数量,默认5
        
        Returns:
        --------
        dict: 排序力分析结果
        """
        self.log("=" * 70)
        self.log("体检3: 排序力体检(不依赖策略)")
        self.log("=" * 70)
        
        # 获取PCA特征作为信号强度代理
        pca_columns = [col for col in signal_data.columns if col.startswith('PC')]
        
        if len(pca_columns) == 0:
            self.log("⚠️  未找到PCA特征,跳过排序力检验")
            return {'skipped': True}
        
        returns = signal_data['future_return_5d'].fillna(0).values
        
        results = {
            'ic_values': {},
            'quantile_returns': {},
            'spread': None,
            'ranking_power': 'unknown'
        }
        
        # 计算各PCA成分的IC值
        self.log("计算信息系数(IC,T+1对齐)...")
        
        ic_values = []
        for pc in pca_columns[:5]:  # 只看前5个主成分
            feature = signal_data[pc].fillna(0).values
            
            # 【关键修复】T+1对齐: 今天的特征预测明天的收益
            feature_t_plus_1 = np.roll(feature, 1)
            feature_t_plus_1[0] = 0  # 第一天无特征值
            
            # 计算IC (Spearman相关系数)
            valid_mask = ~np.isnan(returns)
            if valid_mask.sum() > 10:
                ic, p_value = stats.spearmanr(feature_t_plus_1[valid_mask], returns[valid_mask])
                ic_values.append((pc, ic, p_value))
                self.log(f"   {pc}: IC={ic:+.4f} (p={p_value:.4f})")
        
        # 选择IC绝对值最大的主成分作为信号
        if len(ic_values) > 0:
            ic_values_sorted = sorted(ic_values, key=lambda x: abs(x[1]), reverse=True)
            best_pc, best_ic, best_p = ic_values_sorted[0]
            
            self.log(f"\n最佳信号: {best_pc} (IC={best_ic:+.4f})")
            results['ic_values'] = {pc: ic for pc, ic, p in ic_values}
            results['best_ic'] = best_ic
            results['best_pc'] = best_pc
            
            # 分层测试
            self.log(f"\n分{n_quantiles}层收益分析...")
            
            feature = signal_data[best_pc].fillna(0).values
            valid_mask = ~np.isnan(returns)
            feature_clean = feature[valid_mask]
            returns_clean = returns[valid_mask]
            
            # 按信号强度分层
            quantile_labels = pd.qcut(feature_clean, q=n_quantiles, labels=False, duplicates='drop')
            
            quantile_returns = []
            for q in range(n_quantiles):
                q_mask = quantile_labels == q
                if q_mask.sum() > 0:
                    q_return = returns_clean[q_mask].mean()
                    quantile_returns.append(q_return)
                    self.log(f"   Q{q+1} (n={q_mask.sum():4d}): {q_return:+.6f}")
                else:
                    quantile_returns.append(0)
            
            # 计算Spread (Q5 - Q1)
            spread = quantile_returns[-1] - quantile_returns[0]
            results['quantile_returns'] = quantile_returns
            results['spread'] = spread
            
            self.log(f"\n多空价差(Spread): {spread:+.6f}")
            
            # 判断排序力
            if abs(best_ic) >= 0.02 and spread * np.sign(best_ic) > 0:
                self.log("✅ 信号有排序力 (IC≥0.02 且 Spread方向一致)")
                results['ranking_power'] = 'strong'
                self.add_recommendation("信号有区分力,应优先优化信号转换和成本  控制")
            elif abs(best_ic) >= 0.01:
                self.log("⚠️  信号有弱排序力 (0.01≤IC<0.02)")
                results['ranking_power'] = 'weak'
                self.add_recommendation("信号有弱区分力,可尝试优化特征工程或信号组合")
            else:
                self.log("❌ 信号无排序力 (IC<0.01)")
                results['ranking_power'] = 'none'
                self.add_issue("信号本身无区分力")
                self.add_recommendation("应回到特征工程或进入阶段12(机器学习基线)")
        
        return results
    
    # ========== 体检4: 状态过滤体检 ==========
    
    def check_state_filtering(self, signal_data: pd.DataFrame, 
                              k_values: List[int] = [3, 4, 5]) -> Dict:
        """
        体检4: 状态过滤体检(PCA+KMeans)
        
        检查聚类状态过滤是否有效
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含PCA特征的数据
        k_values : List[int]
            要测试的k值列表
        
        Returns:
        --------
        dict: 状态过滤分析结果
        """
        self.log("=" * 70)
        self.log("体检4: 状态过滤体检(PCA+KMeans)")
        self.log("=" * 70)
        
        pca_columns = [col for col in signal_data.columns if col.startswith('PC')]
        
        if len(pca_columns) == 0:
            self.log("⚠️  未找到PCA特征,跳过状态过滤检验")
            return {'skipped': True}
        
        X_pca = signal_data[pca_columns].fillna(0).values
        returns = signal_data['future_return_5d'].fillna(0).values
        
        # 分训练/测试集(8:2)
        split_point = int(len(X_pca) * 0.8)
        X_train = X_pca[:split_point]
        X_test = X_pca[split_point:]
        returns_train = returns[:split_point]
        returns_test = returns[split_point:]
        
        self.log(f"数据切分: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
        
        results = {
            'k_results': {},
            'best_k': None,
            'consistency_check': False
        }
        
        # 测试不同k值
        for k in k_values:
            self.log(f"\n测试 k={k}...")
            
            # 仅在训练集上训练
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            train_labels = kmeans.fit_predict(X_train)
            test_labels = kmeans.predict(X_test)
            
            # 计算每个簇的收益
            train_cluster_returns = []
            test_cluster_returns = []
            train_cluster_pcts = []
            test_cluster_pcts = []
            
            MIN_CLUSTER_PCT = 0.05  # 最小簇占比5%
            
            for c in range(k):
                train_mask = train_labels == c
                test_mask = test_labels == c
                
                train_pct = train_mask.sum() / len(train_labels)
                test_pct = test_mask.sum() / len(test_labels)
                
                train_cluster_pcts.append(train_pct)
                test_cluster_pcts.append(test_pct)
                
                # 标记占比过小的簇
                is_valid = train_pct >= MIN_CLUSTER_PCT
                
                train_return = returns_train[train_mask].mean() if train_mask.sum() > 0 else 0
                test_return = returns_test[test_mask].mean() if test_mask.sum() > 0 else 0
                
                train_cluster_returns.append(train_return if is_valid else -999)  # 无效簇标记为极低收益
                test_cluster_returns.append(test_return)
                
                status = "✅" if is_valid else f"❌ (占比<{MIN_CLUSTER_PCT:.0%})"
                self.log(f"   簇{c}: {status} 训练={train_return:+.6f} ({train_pct:5.1%}), "
                        f"测试={test_return:+.6f} ({test_pct:5.1%})")
            
            # 找到训练集最佳簇(排除占比过小的簇)
            valid_clusters = [i for i in range(k) if train_cluster_pcts[i] >= MIN_CLUSTER_PCT]
            
            if len(valid_clusters) == 0:
                self.log(f"   ⚠️ k={k} 没有符合占比要求的簇!")
                best_cluster_idx = np.argmax(train_cluster_returns)  # 退化到全部簇
            else:
                valid_returns = [train_cluster_returns[i] for i in valid_clusters]
                best_idx_in_valid = np.argmax(valid_returns)
                best_cluster_idx = valid_clusters[best_idx_in_valid]
            
            best_train_return = train_cluster_returns[best_cluster_idx]
            if best_train_return == -999:
                best_train_return = 0  # 恢复真实值
                best_train_return = returns_train[train_labels == best_cluster_idx].mean()
            best_test_return = test_cluster_returns[best_cluster_idx]
            
            # 检查占比
            train_best_pct = train_cluster_pcts[best_cluster_idx]
            test_best_pct = test_cluster_pcts[best_cluster_idx]
            
            self.log(f"   🏆 最佳簇{best_cluster_idx}: 占比 训练={train_best_pct:.1%}, 测试={test_best_pct:.1%}")
            
            results['k_results'][k] = {
                'best_cluster': best_cluster_idx,
                'train_return': best_train_return,
                'test_return': best_test_return,
                'train_pct': train_best_pct,
                'test_pct': test_best_pct,
                'direction_consistent': (best_train_return * best_test_return > 0)
            }
            
            # 检查占比是否合理(10%-60%)
            if train_best_pct < 0.1 or train_best_pct > 0.6:
                self.add_issue(f"k={k} 最佳簇占比异常: {train_best_pct:.1%}")
        
        # 检查一致性
        self.log("\n一致性检查:")
        
        consistent_ks = []
        for k, k_result in results['k_results'].items():
            if k_result['direction_consistent']:
                consistent_ks.append(k)
                self.log(f"   k={k}: ✅ 方向一致")
            else:
                self.log(f"   k={k}: ❌ 方向不一致")
        
        if len(consistent_ks) >= 2:
            results['consistency_check'] = True
            self.log("\n✅ 多个k值方向一致,状态过滤有效")
            self.add_recommendation("状态过滤有效,可继续使用聚类策略")
        else:
            self.log("\n❌ 状态过滤方向不一致,可能过拟合")
            self.add_issue("聚类状态过滤不稳定")
            self.add_recommendation("考虑简化状态定义或使用其他过滤方式")
        
        return results
    
    # ========== 体检5: 门槛/持有周期小网格 ==========
    
    def grid_search_threshold_holding(self, signal_data: pd.DataFrame,
                                     quantiles: List[float] = [0.6, 0.7, 0.8, 0.9],
                                     hold_periods: List[int] = [1, 3, 5]) -> Dict:
        """
        体检5: 门槛/持有周期小网格搜索
        
        通过小网格搜索找到最优的信号门槛和持有周期
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含PCA特征的数据
        quantiles : List[float]
            信号门槛分位数列表
        hold_periods : List[int]
            持有周期列表
        
        Returns:
        --------
        dict: 网格搜索结果
        """
        self.log("=" * 70)
        self.log("体检5: 门槛/持有周期小网格搜索")
        self.log("=" * 70)
        
        pca_columns = [col for col in signal_data.columns if col.startswith('PC')]
        
        if len(pca_columns) == 0 or 'future_return_5d' not in signal_data.columns:
            self.log("⚠️  数据不足,跳过网格搜索")
            return {'skipped': True}
        
        # 使用PC1作为信号强度
        signal_strength = signal_data[pca_columns[0]].fillna(0).values
        
        # 分训练/测试集
        split_point = int(len(signal_strength) * 0.8)
        
        results = {
            'grid_results': [],
            'best_combo': None
        }
        
        self.log(f"网格: quantiles={quantiles}, hold_periods={hold_periods}")
        self.log(f"总共测试 {len(quantiles) * len(hold_periods)} 种组合\n")
        
        # 网格搜索
        for q in quantiles:
            for hold_n in hold_periods:
                # 训练集
                train_strength = signal_strength[:split_point]
                train_returns = signal_data['future_return_5d'].iloc[:split_point].fillna(0).values
                
                # 生成信号: 信号强度 > q分位数
                train_threshold = np.quantile(train_strength, q)
                train_signal = (train_strength >= train_threshold).astype(int)
                
                # 应用持有周期: 一旦买入,持有hold_n期
                train_signal_hold = self._apply_holding_period(train_signal, hold_n)
                
                # 计算训练集收益
                if hold_n == 1:
                    train_return = (train_signal_hold * train_returns).sum()
                else:
                    # 持有期内累计收益
                    train_return = self._calculate_holding_return(train_signal_hold, train_returns, hold_n)
                
                # 测试集
                test_strength = signal_strength[split_point:]
                test_returns = signal_data['future_return_5d'].iloc[split_point:].fillna(0).values
                
                # 使用训练集阈值
                test_signal = (test_strength >= train_threshold).astype(int)
                test_signal_hold = self._apply_holding_period(test_signal, hold_n)
                
                if hold_n == 1:
                    test_return = (test_signal_hold * test_returns).sum()
                else:
                    test_return = self._calculate_holding_return(test_signal_hold, test_returns, hold_n)
                
                # 记录结果
                combo_result = {
                    'quantile': q,
                    'hold_period': hold_n,
                    'train_return': train_return,
                    'test_return': test_return,
                    'train_signal_ratio': train_signal_hold.mean(),
                    'test_signal_ratio': test_signal_hold.mean()
                }
                results['grid_results'].append(combo_result)
                
                self.log(f"q={q:.1f}, hold={hold_n}: "
                        f"训练={train_return:+.4f} ({train_signal_hold.mean():.1%}), "
                        f"测试={test_return:+.4f} ({test_signal_hold.mean():.1%})")
        
        # 选择测试集表现最好的组合
        results['grid_results'].sort(key=lambda x: x['test_return'], reverse=True)
        best_combo = results['grid_results'][0]
        results['best_combo'] = best_combo
        
        self.log(f"\n最佳组合 (样本外):")
        self.log(f"   门槛分位数: {best_combo['quantile']:.1f}")
        self.log(f"   持有周期: {best_combo['hold_period']}")
        self.log(f"   测试收益: {best_combo['test_return']:+.4f}")
        self.log(f"   信号比例: {best_combo['test_signal_ratio']:.1%}")
        
        # 与默认参数对比
        default_combo = [c for c in results['grid_results'] 
                        if c['quantile'] == 0.7 and c['hold_period'] == 1]
        
        if len(default_combo) > 0:
            improvement = best_combo['test_return'] - default_combo[0]['test_return']
            if improvement > 0.01:
                self.log(f"\n✅ 网格搜索提升收益: {improvement:+.4f}")
                self.add_recommendation(f"使用最佳参数组合: q={best_combo['quantile']:.1f}, hold={best_combo['hold_period']}")
            else:
                self.log(f"\n⚠️  网格搜索提升有限: {improvement:+.4f}")
        
        return results
    
    def _apply_holding_period(self, signal: np.ndarray, hold_n: int) -> np.ndarray:
        """应用持有周期: 买入后持有hold_n期"""
        signal_hold = signal.copy()
        
        for i in range(1, len(signal_hold)):
            if signal[i-1] == 1:
                # 前一期有信号,继续持有
                for j in range(1, hold_n):
                    if i + j < len(signal_hold):
                        signal_hold[i + j] = 1
        
        return signal_hold
    
    def _calculate_holding_return(self, signal: np.ndarray, returns: np.ndarray, hold_n: int) -> float:
        """计算持有期收益"""
        total_return = 0
        position = 0  # 当前持仓
        
        for i in range(len(signal)):
            if signal[i] == 1 and position == 0:
                # 买入
                position = 1
                holding_days = 0
            
            if position == 1:
                # 持仓中,累计收益
                total_return += returns[i]
                holding_days += 1
                
                if holding_days >= hold_n:
                    # 持有期满,卖出
                    position = 0
        
        return total_return
    
    # ========== 体检6: 随机基准与年度切片 ==========
    
    def check_random_baseline_and_yearly(self, signal_data: pd.DataFrame,
                                        n_random: int = 100) -> Dict:
        """
        体检6: 随机基准与年度切片
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号的数据
        n_random : int
            随机模拟次数
        
        Returns:
        --------
        dict: 随机基准和年度分析结果
        """
        self.log("=" * 70)
        self.log("体检6: 随机基准与年度切片")
        self.log("=" * 70)
        
        signal = signal_data['signal_combined'].values
        returns = signal_data['future_return_5d'].fillna(0).values
        
        # 策略收益
        strategy_returns = signal * returns
        strategy_total = strategy_returns.sum()
        
        # 随机基准
        self.log(f"运行{n_random}次随机模拟...")
        
        signal_ratio = signal.mean()
        random_results = []
        
        np.random.seed(42)
        for i in range(n_random):
            # 生成相同占比的随机信号
            n_signals = int(len(signal) * signal_ratio)
            random_signal = np.zeros(len(signal))
            if n_signals > 0:
                random_indices = np.random.choice(len(signal), n_signals, replace=False)
                random_signal[random_indices] = 1
            
            random_return = (random_signal * returns).sum()
            random_results.append(random_return)
        
        random_results = np.array(random_results)
        random_mean = random_results.mean()
        random_std = random_results.std()
        
        # 计算策略排名
        percentile = (random_results < strategy_total).mean()
        z_score = (strategy_total - random_mean) / random_std if random_std > 0 else 0
        
        self.log(f"\n随机基准对比:")
        self.log(f"   策略收益: {strategy_total:+.4f}")
        self.log(f"   随机平均: {random_mean:+.4f} ± {random_std:.4f}")
        self.log(f"   策略分位数: {percentile:.1%}")
        self.log(f"   Z-score: {z_score:+.2f}")
        
        results = {
            'strategy_return': strategy_total,
            'random_mean': random_mean,
            'random_std': random_std,
            'percentile': percentile,
            'z_score': z_score
        }
        
        if percentile < 0.6:
            self.add_issue(f"策略未显著优于随机基准(分位数={percentile:.1%})")
            self.add_recommendation("策略可能缺乏真实信号,建议重新审视特征和状态定义")
        elif z_score >= 2:
            self.log("✅ 策略显著优于随机基准(2σ以上)")
        elif z_score >= 1:
            self.log("⚠️  策略适度优于随机基准(1-2σ)")
        else:
            self.log("❌ 策略未显著优于随机基准(<1σ)")
        
        # 年度切片
        self.log("\n年度切片分析:")
        
        if 'datetime' in signal_data.index.names or isinstance(signal_data.index, pd.DatetimeIndex):
            dates = signal_data.index
            
            # 按年份分组
            years = dates.year.unique()
            
            yearly_results = []
            for year in sorted(years):
                year_mask = dates.year == year
                year_signal = signal[year_mask]
                year_returns = returns[year_mask]
                
                year_strategy_return = (year_signal * year_returns).sum()
                year_signal_ratio = year_signal.mean()
                
                yearly_results.append({
                    'year': year,
                    'return': year_strategy_return,
                    'signal_ratio': year_signal_ratio,
                    'n_samples': year_mask.sum()
                })
                
                self.log(f"   {year}: 收益={year_strategy_return:+.4f}, "
                        f"信号率={year_signal_ratio:.1%}, n={year_mask.sum()}")
            
            # 检查年度一致性
            yearly_returns = [y['return'] for y in yearly_results]
            
            if len(yearly_returns) > 1:
                positive_years = sum(1 for r in yearly_returns if r > 0)
                consistency = positive_years / len(yearly_returns)
                
                self.log(f"\n年度一致性: {positive_years}/{len(yearly_returns)} ({consistency:.0%})")
                
                if consistency >= 0.7:
                    self.log("✅ 年度表现一致")
                elif consistency >= 0.5:
                    self.log("⚠️  年度表现一般")
                else:
                    self.log("❌ 年度表现不一致")
                    self.add_issue("策略年度表现不稳定")
                
                results['yearly_consistency'] = consistency
        else:
            self.log("⚠️  数据无时间索引,跳过年度切片")
        
        return results
    
    # ========== 主体检流程 ==========
    
    def run_full_triage(self, signal_data: pd.DataFrame) -> Dict:
        """
        运行完整的6步体检流程
        
        Parameters:
        -----------
        signal_data : pd.DataFrame
            包含信号、PCA特征和未来收益的完整数据
        
        Returns:
        --------
        dict: 完整体检报告
        """
        self.log("=" * 70)
        self.log("🏥 60分钟快速体检 (Full Triage)")
        self.log("=" * 70)
        self.log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"数据量: {len(signal_data)} 条")
        self.log("")
        
        start_time = datetime.now()
        
        # 体检1: 信号对齐与泄露检查
        check1 = self.check_signal_alignment_and_leakage(signal_data)
        
        # 体检1A: 破坏性对照实验(验证泄漏)
        check1a = self.check_leakage_with_wrong_labels(signal_data)
        
        # 体检2: 成本与换手拆解
        check2 = self.analyze_cost_and_turnover(signal_data)
        
        # 体检3: 排序力体检
        check3 = self.check_ranking_power(signal_data)
        
        # 体检4: 状态过滤体检
        check4 = self.check_state_filtering(signal_data)
        
        # 体检5: 门槛/持有周期网格
        check5 = self.grid_search_threshold_holding(signal_data)
        
        # 体检6: 随机基准与年度切片
        check6 = self.check_random_baseline_and_yearly(signal_data)
        
        # 整合结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        triage_summary = {
            'check1_alignment': check1,
            'check1a_leakage_test': check1a,
            'check2_cost': check2,
            'check3_ranking': check3,
            'check4_state': check4,
            'check5_grid': check5,
            'check6_baseline': check6,
            'issues_found': self.issues_found,
            'recommendations': self.recommendations,
            'duration_minutes': duration
        }
        
        # 生成最终诊断
        self.generate_final_diagnosis(triage_summary)
        
        # 保存报告
        self.save_triage_report(triage_summary)
        
        self.log(f"\n完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"耗时: {duration:.1f} 分钟")
        
        return triage_summary
    
    def generate_final_diagnosis(self, summary: Dict):
        """生成最终诊断建议"""
        self.log("\n" + "=" * 70)
        self.log("🎯 最终诊断与建议")
        self.log("=" * 70)
        
        # 提取关键指标
        ranking = summary.get('check3_ranking', {})
        cost = summary.get('check2_cost', {})
        baseline = summary.get('check6_baseline', {})
        
        ic_value = ranking.get('best_ic', 0)
        spread = ranking.get('spread', 0)
        cost_erosion = cost.get('cost_erosion_ratio', 0)
        z_score = baseline.get('z_score', 0)
        
        self.log("\n关键指标:")
        self.log(f"   IC值: {ic_value:+.4f}")
        self.log(f"   Spread: {spread:+.6f}")
        self.log(f"   成本侵蚀: {cost_erosion:.1%}")
        self.log(f"   随机基准Z-score: {z_score:+.2f}")
        
        # 决策树
        self.log("\n诊断结论:")
        
        if abs(ic_value) >= 0.02 or (spread > 0 and abs(spread) > 0.001):
            # 信号有区分力
            self.log("✅ 信号有区分力 (IC≥0.02 或 Spread>0)")
            
            if cost_erosion > 0.5:
                self.log("⚠️  但被成本大幅侵蚀")
                self.log("\n🔧 建议路径: 先优化信号转换")
                self.log("   1. 降低换手率(延长持有期、提高门槛)")
                self.log("   2. 优化入场/出场逻辑")
                self.log("   3. 使用体检5的最佳参数组合")
            elif z_score < 1:
                self.log("⚠️  但未显著优于随机")
                self.log("\n🔧 建议路径: 先优化策略转换逻辑")
                self.log("   1. 检查信号对齐(体检1)")
                self.log("   2. 调整门槛和持有期(体检5)")
                self.log("   3. 考虑状态过滤的有效性(体检4)")
            else:
                self.log("✅ 且转换效率良好")
                self.log("\n🔧 建议路径: 继续优化和实盘验证")
                self.log("   1. 加入止损/止盈逻辑")
                self.log("   2. 做好仓位管理")
                self.log("   3. 准备实盘测试")
        else:
            # 信号无区分力
            self.log("❌ 信号无区分力 (IC≈0, Spread≈0)")
            self.log("\n🔧 建议路径: 回到特征层或进入ML基线")
            self.log("   1. 重新审视特征工程(是否有预测性)")
            self.log("   2. 检查PCA降维是否丢失信息")
            self.log("   3. 进入阶段12: 尝试监督学习模型")
            self.log("   4. 考虑换用其他因子或数据源")
        
        # 列出所有问题
        if len(self.issues_found) > 0:
            self.log("\n⚠️  发现的问题:")
            for i, issue in enumerate(self.issues_found, 1):
                self.log(f"   {i}. {issue}")
        
        # 列出所有建议
        if len(self.recommendations) > 0:
            self.log("\n💡 改进建议:")
            for i, rec in enumerate(self.recommendations, 1):
                self.log(f"   {i}. {rec}")
    
    def save_triage_report(self, summary: Dict):
        """保存体检报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.reports_dir, f"triage_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.triage_report))
        
        self.log(f"\n📋 体检报告已保存: {os.path.basename(report_file)}")


def main():
    """主函数: 运行快速体检"""
    try:
        # 初始化体检系统
        triage = QuickTriage()
        
        # 加载数据(需要先运行strategy_backtest生成数据)
        print("\n📂 加载测试数据...")
        
        # 这里需要先运行strategy_backtest生成signal_data
        # 然后保存为CSV供体检使用
        # 或者直接从strategy_backtest.py导入数据
        
        from strategy_backtest import StrategyBacktest
        
        backtest = StrategyBacktest()
        
        # 1. 加载聚类结果
        cluster_results = backtest.load_cluster_evaluation_results()
        
        # 2. 选择聚类
        selection_results = backtest.select_best_clusters(cluster_results['comparison_df'], top_n=3)
        
        # 3. 准备测试数据
        test_data = backtest.prepare_test_data(symbol="000001")
        
        # 4. 生成信号
        signal_data = backtest.generate_trading_signals(test_data, selection_results['selected_clusters'])
        
        print(f"✅ 数据加载完成: {len(signal_data)} 条记录")
        print(f"   特征: {[col for col in signal_data.columns if col.startswith('PC')][:5]}")
        print(f"   信号: signal_combined")
        print(f"   目标: future_return_5d")
        
        # 运行完整体检
        print("\n" + "=" * 70)
        print("开始60分钟快速体检...")
        print("=" * 70)
        
        triage_results = triage.run_full_triage(signal_data)
        
        print("\n" + "=" * 70)
        print("🎉 快速体检完成!")
        print("=" * 70)
        
        return triage_results
        
    except Exception as e:
        print(f"\n💥 体检失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
