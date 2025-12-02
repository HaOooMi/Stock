"""
评估模块

包含：
1. 传统评估指标（metrics, bucketing, reporting）
2. 横截面评估框架（Alphalens风格）
   - cross_section_metrics: 核心度量计算
   - factor_preprocessing: 因子预处理
   - cross_section_analyzer: 分析器主类
   - visualization: 可视化工具
   - tearsheet: 报告生成
"""

# 传统评估模块
from .metrics import calculate_metrics, calculate_ic_by_date
from .bucketing import bucket_predictions, analyze_bucket_performance
from .reporting import generate_report

# 横截面评估框架
from .cross_section_analyzer import CrossSectionAnalyzer
from .cross_section_adapter import CrossSectionAdapter, quick_evaluate
from .cross_section_metrics import (
    calculate_forward_returns,
    calculate_daily_ic,
    calculate_ic_summary,
    calculate_quantile_returns,
    calculate_cumulative_returns,
    calculate_spread,
    calculate_monotonicity,
    calculate_turnover
)
from .factor_preprocessing import (
    winsorize_factor,
    standardize_factor,
    neutralize_factor,
    preprocess_factor_pipeline
)
from .visualization import (
    plot_ic_time_series,
    plot_ic_distribution,
    plot_quantile_cumulative_returns,
    plot_quantile_mean_returns,
    plot_spread_cumulative_returns,
    plot_turnover_time_series,
    plot_monthly_ic_heatmap,
    create_factor_tearsheet_plots
)
from .tearsheet import (
    generate_html_tearsheet,
    generate_full_tearsheet,
    save_ic_to_csv,
    save_quantile_returns_to_csv
)
from .drift_detector import (
    DriftDetector,
    compare_splits_with_analyzer
)

__all__ = [
    # 传统模块
    'calculate_metrics',
    'calculate_ic_by_date',
    'bucket_predictions',
    'analyze_bucket_performance',
    'generate_report',
    
    # 横截面评估框架
    'CrossSectionAnalyzer',
    'CrossSectionAdapter',
    'quick_evaluate',
    
    # 核心度量
    'calculate_forward_returns',
    'calculate_daily_ic',
    'calculate_ic_summary',
    'calculate_quantile_returns',
    'calculate_cumulative_returns',
    'calculate_spread',
    'calculate_monotonicity',
    'calculate_turnover',
    
    # 因子预处理
    'winsorize_factor',
    'standardize_factor',
    'neutralize_factor',
    'preprocess_factor_pipeline',
    
    # 可视化
    'plot_ic_time_series',
    'plot_ic_distribution',
    'plot_quantile_cumulative_returns',
    'plot_quantile_mean_returns',
    'plot_spread_cumulative_returns',
    'plot_turnover_time_series',
    'plot_monthly_ic_heatmap',
    'create_factor_tearsheet_plots',
    
    # 报告生成
    'generate_html_tearsheet',
    'generate_full_tearsheet',
    'save_ic_to_csv',
    'save_quantile_returns_to_csv',
    
    # 漂移检测
    'DriftDetector',
    'compare_splits_with_analyzer'
]
