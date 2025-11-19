#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面评估适配器 - 对接现有DataLoader和训练流程

功能：
1. 从 DataLoader 输出自动提取因子、价格、元数据
2. 与 train_models.py 无缝集成
3. 支持单股票时序 和 多股票横截面 两种模式
4. 自动从 InfluxDB/CSV 加载 prices 数据
5. 提供一键评估接口

适配对象：
- data/data_loader.py
- data/market_data_loader.py  
- pipelines/train_models.py
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(current_dir)
project_root = os.path.dirname(ml_root)
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入现有模块
from data.data_loader import DataLoader
from data.market_data_loader import MarketDataLoader

# 导入股票元数据模块
try:
    from get_stock_info.stock_meta_akshare import get_basic_info_mysql
    from sqlalchemy import create_engine
    HAVE_STOCK_META = True
except ImportError:
    HAVE_STOCK_META = False
    print("⚠️ 股票元数据模块未找到")

# 导入评估核心模块
try:
    from evaluation.cross_section_analyzer import CrossSectionAnalyzer
    from evaluation.tearsheet import generate_full_tearsheet
    HAVE_CROSS_SECTION = True
except ImportError:
    HAVE_CROSS_SECTION = False
    print("⚠️ 横截面评估模块未找到")


class CrossSectionAdapter:
    """
    横截面评估适配器
    
    功能：
    1. 自动从 DataLoader 提取数据
    2. 从 MarketDataLoader 获取 prices
    3. 提供一键评估接口
    4. 支持单股票和多股票模式
    """
    
    def __init__(self, 
                 data_loader: DataLoader,
                 market_data_loader: Optional[MarketDataLoader] = None,
                 enable_neutralization: bool = False,
                 db_engine = None):
        """
        初始化适配器
        
        Parameters:
        -----------
        data_loader : DataLoader
            现有的数据加载器实例
        market_data_loader : MarketDataLoader, optional
            市场数据加载器（用于获取prices）
        enable_neutralization : bool
            是否启用市值/行业中性化（仅多股票模式）
        db_engine : sqlalchemy.Engine, optional
            MySQL数据库引擎（用于获取市值和行业数据）
        """
        self.data_loader = data_loader
        self.market_data_loader = market_data_loader
        self.enable_neutralization = enable_neutralization
        self.db_engine = db_engine
        
        if not HAVE_CROSS_SECTION:
            raise ImportError("请先实现 cross_section_analyzer.py 和 tearsheet.py")
        
        print(f"🔌 横截面评估适配器初始化")
        print(f"   数据加载器: ✅")
        print(f"   市场数据: {'✅' if market_data_loader else '❌'}")
        print(f"   数据库连接: {'✅' if db_engine else '❌'}")
        print(f"   中性化: {'✅' if enable_neutralization else '❌'}")
    
    def evaluate_feature(self,
                        features: pd.DataFrame,
                        targets: pd.Series,
                        feature_col: str,
                        symbol: str,
                        start_date: str,
                        end_date: str,
                        forward_periods: List[int] = [5],
                        quantiles: int = 5,
                        output_dir: Optional[str] = None) -> Dict:
        """
        评估单个特征的预测能力（对接 DataLoader 输出）
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataLoader 返回的特征数据（MultiIndex[date, ticker]）
        targets : pd.Series
            DataLoader 返回的目标数据
        feature_col : str
            要评估的特征列名
        symbol : str
            股票代码（用于加载prices）
        start_date : str
            开始日期
        end_date : str
            结束日期
        forward_periods : List[int]
            向前期数（天）
        quantiles : int
            分位数数量
        output_dir : str, optional
            输出目录（默认：ML output/reports/baseline_v1/factors）
            
        Returns:
        --------
        Dict
            评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"📊 评估特征: {feature_col}")
        print(f"{'='*60}")
        
        # 1. 提取因子数据（单列）
        if feature_col not in features.columns:
            raise ValueError(f"特征列 '{feature_col}' 不存在于features中")
        
        factor_df = features[[feature_col]].copy()
        
        # 2. 检测是否为单股票场景
        n_symbols = factor_df.index.get_level_values('ticker').nunique()
        is_single_stock = (n_symbols == 1)
        
        print(f"\n   📈 数据信息:")
        print(f"      股票数量: {n_symbols} ({'单股票' if is_single_stock else '多股票'})")
        print(f"      样本数量: {len(factor_df)}")
        print(f"      时间范围: {factor_df.index.get_level_values('date').min().date()} ~ "
              f"{factor_df.index.get_level_values('date').max().date()}")
        
        # 3. 获取 prices 数据
        prices_df = self._load_prices(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            index=factor_df.index
        )
        
        # 4. 准备可选数据（仅多股票模式）
        market_cap_df = None
        industry_df = None
        
        if not is_single_stock and self.enable_neutralization:
            print(f"\n   📊 加载市值和行业数据用于中性化...")
            market_cap_df, industry_df = self._load_market_cap_and_industry(
                index=factor_df.index
            )
        
        # 5. 创建 CrossSectionAnalyzer
        analyzer = CrossSectionAnalyzer(
            factors=factor_df,
            prices=prices_df,
            market_cap=market_cap_df,
            industry=industry_df,
            tradable_mask=None,  # DataLoader 已过滤
            forward_periods=forward_periods,
            quantiles=quantiles,
            return_type='simple'
        )
        
        # 6. 预处理（横截面标准化）
        analyzer.preprocess(
            winsorize=True,
            standardize=True,
            neutralize=(not is_single_stock and self.enable_neutralization)  # 多股票且启用中性化
        )
        
        # 7. 执行分析
        results = analyzer.analyze()
        
        # 8. 打印摘要
        analyzer.summary()
        
        # 9. 生成报告（如果指定输出目录）
        if output_dir:
            # 确定输出目录
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(ml_root, output_dir)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成完整报告
            generate_full_tearsheet(
                results,
                factor_name=feature_col,
                output_dir=output_dir,
                show_plots=False
            )
            
            print(f"\n   ✅ 报告已生成: {output_dir}")
        
        return results
    
    def evaluate_all_features(self,
                             features: pd.DataFrame,
                             targets: pd.Series,
                             symbol: str,
                             start_date: str,
                             end_date: str,
                             output_dir: Optional[str] = None,
                             top_k: Optional[int] = None) -> pd.DataFrame:
        """
        批量评估所有特征（对接 DataLoader 输出）
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataLoader 返回的特征数据
        targets : pd.Series
            DataLoader 返回的目标数据
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
        output_dir : str, optional
            输出目录
        top_k : int, optional
            仅评估前K个特征
            
        Returns:
        --------
        pd.DataFrame
            特征评估汇总表（IC、ICIR、IC胜率等）
        """
        print(f"\n{'='*60}")
        print(f"📊 批量评估所有特征")
        print(f"{'='*60}")
        
        feature_cols = features.columns.tolist()
        if top_k:
            feature_cols = feature_cols[:top_k]
        
        print(f"\n   待评估特征数: {len(feature_cols)}")
        
        # 加载prices（共享）
        prices_df = self._load_prices(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            index=features.index
        )
        
        # 批量评估
        summary_list = []
        
        for i, feature_col in enumerate(feature_cols, 1):
            print(f"\n[{i}/{len(feature_cols)}] 评估: {feature_col}")
            
            try:
                # 评估单个特征
                results = self._evaluate_single_feature(
                    features[[feature_col]],
                    targets,
                    prices_df,
                    feature_col
                )
                
                # 提取关键指标
                ic_summary = results.get('ic_summary_5', {})
                
                summary_list.append({
                    'feature': feature_col,
                    'ic_mean': ic_summary.get('ic_mean', np.nan),
                    'ic_std': ic_summary.get('ic_std', np.nan),
                    'icir': ic_summary.get('ic_ir', np.nan),
                    'icir_annual': ic_summary.get('ic_ir_annual', np.nan),
                    'ic_win_rate': ic_summary.get('ic_win_rate', np.nan),
                    'p_value': ic_summary.get('p_value', np.nan),
                    't_stat': ic_summary.get('t_stat', np.nan),
                    'qualified': (ic_summary.get('ic_ir', 0) > 0.5 and 
                                 ic_summary.get('p_value', 1) < 0.05)
                })
                
                # 快速反馈
                if summary_list[-1]['qualified']:
                    print(f"   ✅ 合格特征 (IC={summary_list[-1]['ic_mean']:.4f}, "
                          f"ICIR={summary_list[-1]['icir']:.2f})")
                else:
                    print(f"   ❌ 弱特征 (IC={summary_list[-1]['ic_mean']:.4f}, "
                          f"ICIR={summary_list[-1]['icir']:.2f})")
                
            except Exception as e:
                print(f"   ⚠️  评估失败: {e}")
                summary_list.append({
                    'feature': feature_col,
                    'ic_mean': np.nan,
                    'ic_std': np.nan,
                    'icir': np.nan,
                    'icir_annual': np.nan,
                    'ic_win_rate': np.nan,
                    'p_value': np.nan,
                    't_stat': np.nan,
                    'qualified': False
                })
        
        # 汇总结果
        summary_df = pd.DataFrame(summary_list)
        summary_df = summary_df.sort_values('icir', ascending=False)
        
        # 打印TOP特征
        print(f"\n{'='*60}")
        print(f"📈 TOP 10 特征（按ICIR排序）")
        print(f"{'='*60}")
        print(summary_df.head(10).to_string(index=False))
        
        # 保存汇总表
        if output_dir:
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(ml_root, output_dir)
            
            os.makedirs(output_dir, exist_ok=True)
            summary_path = os.path.join(output_dir, 'feature_evaluation_summary.csv')
            summary_df.to_csv(summary_path, index=False, encoding='utf-8')
            print(f"\n   ✅ 汇总表已保存: {summary_path}")
        
        return summary_df
    
    def _evaluate_single_feature(self,
                                 factor_df: pd.DataFrame,
                                 targets: pd.Series,
                                 prices_df: pd.DataFrame,
                                 feature_name: str) -> Dict:
        """
        评估单个特征（内部方法，不生成报告）
        
        Parameters:
        -----------
        factor_df : pd.DataFrame
            单列因子数据
        targets : pd.Series
            目标数据
        prices_df : pd.DataFrame
            价格数据
        feature_name : str
            特征名称
            
        Returns:
        --------
        Dict
            评估结果
        """
        analyzer = CrossSectionAnalyzer(
            factors=factor_df,
            prices=prices_df,
            market_cap=None,
            industry=None,
            tradable_mask=None,
            forward_periods=[5],
            quantiles=5,
            return_type='simple'
        )
        
        analyzer.preprocess(
            winsorize=True,
            standardize=True,
            neutralize=False
        )
        
        results = analyzer.analyze()
        
        return results
    
    def _load_prices(self,
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    index: pd.MultiIndex) -> pd.DataFrame:
        """
        加载价格数据（从 MarketDataLoader 或 features 中提取）
        
        Parameters:
        -----------
        symbol : str
            股票代码
        start_date : str
            开始日期
        end_date : str
            结束日期
        index : pd.MultiIndex
            目标索引（用于对齐）
            
        Returns:
        --------
        pd.DataFrame
            价格数据（MultiIndex[date, ticker]，包含'close'列）
        """
        print(f"\n   📊 加载价格数据...")
        
        # 方式1: 从 MarketDataLoader 加载（推荐）
        if self.market_data_loader is not None:
            try:
                market_df = self.market_data_loader.load_market_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not market_df.empty and 'close' in market_df.columns:
                    # 对齐到目标索引
                    prices_df = market_df[['close']].reindex(index)
                    
                    print(f"      ✅ 从 MarketDataLoader 加载: {len(prices_df)} 行")
                    return prices_df
                    
            except Exception as e:
                print(f"      ⚠️  MarketDataLoader 加载失败: {e}")
        
        # 方式2: 从 DataLoader 的原始文件中提取（备用）
        try:
            data_root = self.data_loader.data_root
            target_files = [f for f in os.listdir(data_root) 
                          if f.startswith(f"with_targets_{symbol}_complete_")]
            
            if target_files:
                target_files.sort(reverse=True)
                target_file = os.path.join(data_root, target_files[0])
                
                df = pd.read_csv(target_file, index_col=0, parse_dates=True)
                
                if 'close' in df.columns:
                    # 转换为 MultiIndex
                    dates = df.index
                    tickers = [symbol] * len(dates)
                    multi_index = pd.MultiIndex.from_arrays(
                        [dates, tickers], 
                        names=['date', 'ticker']
                    )
                    
                    prices_df = pd.DataFrame({
                        'close': df['close'].values
                    }, index=multi_index)
                    
                    # 对齐到目标索引
                    prices_df = prices_df.reindex(index)
                    
                    print(f"      ✅ 从 CSV 文件提取: {len(prices_df)} 行")
                    return prices_df
                    
        except Exception as e:
            print(f"      ⚠️  CSV 文件提取失败: {e}")
        
        # 方式3: 使用目标数据反推（最后备选）
        print(f"      ⚠️  无法加载价格数据，将使用目标数据估算")
        
        # 注意：这种方式有前视偏差，仅用于测试
        # future_return = (price_future - price_now) / price_now
        # => price_now = price_future / (1 + future_return)
        
        # 这里我们无法准确估算，返回空DataFrame
        prices_df = pd.DataFrame({'close': np.nan}, index=index)
        
        return prices_df
    
    def _load_market_cap_and_industry(self,
                                      index: pd.MultiIndex) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        从MySQL数据库加载市值和行业数据
        
        Parameters:
        -----------
        index : pd.MultiIndex
            目标索引（date, ticker）
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (市值数据, 行业数据)，均为MultiIndex[date, ticker]
        """
        if not HAVE_STOCK_META or self.db_engine is None:
            print(f"      ⚠️  无法加载市值和行业数据：缺少数据库连接或股票元数据模块")
            return None, None
        
        try:
            # 提取所有唯一股票代码
            tickers = index.get_level_values('ticker').unique().tolist()
            
            # 从MySQL获取股票基本信息
            with self.db_engine.connect() as conn:
                stock_info = get_basic_info_mysql(conn, tuple(tickers))
            
            if not stock_info:
                print(f"      ⚠️  未找到股票基本信息")
                return None, None
            
            # 构建市值DataFrame（使用流通市值）
            market_cap_data = []
            industry_data = []
            
            for (date, ticker) in index:
                if ticker in stock_info:
                    info = stock_info[ticker]
                    # 市值（使用流通市值，单位：元）
                    market_cap = info.get('流通市值')
                    if pd.notna(market_cap):
                        market_cap_data.append({
                            'date': date,
                            'ticker': ticker,
                            'market_cap': float(market_cap)
                        })
                    
                    # 行业
                    industry = info.get('所属行业')
                    if pd.notna(industry):
                        industry_data.append({
                            'date': date,
                            'ticker': ticker,
                            'industry': str(industry)
                        })
            
            # 转换为DataFrame
            if market_cap_data:
                market_cap_df = pd.DataFrame(market_cap_data)
                market_cap_df = market_cap_df.set_index(['date', 'ticker'])
                print(f"      ✅ 加载市值数据: {len(market_cap_df)} 行")
            else:
                market_cap_df = None
                print(f"      ⚠️  未找到市值数据")
            
            if industry_data:
                industry_df = pd.DataFrame(industry_data)
                industry_df = industry_df.set_index(['date', 'ticker'])
                print(f"      ✅ 加载行业数据: {len(industry_df)} 行")
            else:
                industry_df = None
                print(f"      ⚠️  未找到行业数据")
            
            return market_cap_df, industry_df
            
        except Exception as e:
            print(f"      ⚠️  加载市值和行业数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def quick_evaluate(symbol: str,
                  feature_col: str,
                  data_root: str = "ML output/datasets/baseline_v1",
                  target_col: str = 'future_return_5d',
                  use_scaled: bool = True,
                  output_dir: Optional[str] = None,
                  enable_neutralization: bool = False,
                  db_config: Optional[Dict] = None) -> Dict:
    """
    快速评估接口（一键调用）
    
    Parameters:
    -----------
    symbol : str
        股票代码
    feature_col : str
        要评估的特征列
    data_root : str
        数据根目录
    target_col : str
        目标列名
    use_scaled : bool
        是否使用标准化特征
    output_dir : str, optional
        输出目录
    enable_neutralization : bool
        是否启用市值/行业中性化（仅多股票模式）
    db_config : dict, optional
        数据库配置，格式: {'host': 'localhost', 'user': 'root', 'password': 'xxx', 'database': 'stock_data'}
        
    Returns:
    --------
    Dict
        评估结果
    """
    # 初始化 DataLoader
    data_loader = DataLoader(
        data_root=data_root,
        enable_snapshot=False,
        enable_filtering=False,
        enable_pit_alignment=False,
        enable_influxdb=False
    )
    
    # 加载数据
    features, targets = data_loader.load_features_and_targets(
        symbol=symbol,
        target_col=target_col,
        use_scaled=use_scaled
    )
    
    # 提取日期范围
    dates = features.index.get_level_values('date')
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = dates.max().strftime('%Y-%m-%d')
    
    # 创建数据库引擎（如果提供了配置）
    db_engine = None
    if enable_neutralization and db_config and HAVE_STOCK_META:
        try:
            db_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
            db_engine = create_engine(db_url)
            print(f"✅ 数据库连接成功")
        except Exception as e:
            print(f"⚠️ 数据库连接失败: {e}")
    
    # 初始化适配器
    adapter = CrossSectionAdapter(
        data_loader=data_loader,
        market_data_loader=None,
        enable_neutralization=enable_neutralization,
        db_engine=db_engine
    )
    
    # 执行评估
    results = adapter.evaluate_feature(
        features=features,
        targets=targets,
        feature_col=feature_col,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    return results


if __name__ == "__main__":
    """
    使用示例
    """
    print("=" * 60)
    print("🧪 横截面评估适配器测试")
    print("=" * 60)
    
    # 快速评估示例
    symbol = "000001"
    feature_col = "volume"  # 替换为你的特征列名
    
    try:
        results = quick_evaluate(
            symbol=symbol,
            feature_col=feature_col,
            output_dir="ML output/reports/baseline_v1/factors"
        )
        
        print(f"\n✅ 评估完成")
        print(f"   IC均值: {results['ic_summary_5']['ic_mean']:.4f}")
        print(f"   ICIR: {results['ic_summary_5']['ic_ir']:.2f}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
