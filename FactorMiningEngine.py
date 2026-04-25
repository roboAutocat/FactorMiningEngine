"""
金融因子挖掘系统 - 基于随机字母组合的因子发现与验证
"""

import random
import string
import time
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from scipy import stats
import requests
from bs4 import BeautifulSoup

# 设置中文显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


# ==================== 1. 随机字母生成与词汇检索模块 ====================

class VocabularyGenerator:
    """随机字母生成与金融/新闻词汇检索"""
    
    def __init__(self):
        # 金融专业术语词典（拼音首字母映射）
        self.financial_terms = {
            'a': ['alpha', 'arbitrage', 'asset', 'amplitude', 'anomaly', 'ask', 'auction'],
            'b': ['beta', 'bookvalue', 'bounce', 'bubble', 'bull', 'bear', 'bid', 'backtest'],
            'c': ['carry', 'correlation', 'capex', 'cashflow', 'contrarian', 'capm', 'cds'],
            'd': ['drawdown', 'duration', 'dividend', 'default', 'delta', 'deficit', 'demand'],
            'e': ['earnings', 'equity', 'esg', 'etf', 'exposure', 'expense', 'expectation'],
            'f': ['factor', 'forward', 'future', 'fx', 'fundamental', 'fee', 'friction'],
            'g': ['gamma', 'gdp', 'growth', 'gross', 'gap', 'green', 'gearing'],
            'h': ['hedge', 'hft', 'holding', 'high', 'hurdle', 'haircut', 'housing'],
            'i': ['illiquidity', 'inflation', 'interest', 'intrinsic', 'ipo', 'index', 'impact'],
            'j': ['jensen', 'jump', 'jpy', 'joint', 'judgement', 'jitter'],
            'k': ['kurtosis', 'kline', 'keynes', 'knock', 'knowledge', 'keiretsu'],
            'l': ['leverage', 'liquidity', 'long', 'lstm', 'limit', 'lag', 'loss'],
            'm': ['momentum', 'macro', 'market', 'mean', 'median', 'moneyness', 'margin'],
            'n': ['nav', 'noise', 'normal', 'notional', 'npl', 'net', 'nominal'],
            'o': ['option', 'orderflow', 'overbought', 'oversold', 'open', 'offer', 'outlier'],
            'p': ['pe', 'pb', 'profit', 'portfolio', 'put', 'premium', 'position', 'pair'],
            'q': ['quant', 'quote', 'quality', 'quantity', 'quartile', 'quanto', 'query'],
            'r': ['return', 'risk', 'reversal', 'rebalance', 'roi', 'repo', 'regime'],
            's': ['sharpe', 'size', 'sentiment', 'skew', 'spread', 'swap', 'short', 'style'],
            't': ['turnover', 'trend', 'tail', 'theta', 'technical', 'tracking', 'twap'],
            'u': ['unemployment', 'upside', 'utility', 'underlying', 'unexpected', 'unit'],
            'v': ['volatility', 'value', 'volume', 'var', 'vega', 'valuation', 'variance'],
            'w': ['wacc', 'warrant', 'wavelength', 'wealth', 'window', 'winner', 'wti'],
            'x': ['xirr', 'xsection', 'xrate', 'xenocurrency', 'xmark'],
            'y': ['yield', 'yoy', 'yen', 'year', 'ytd', 'yank'],
            'z': ['zscore', 'zero', 'zone', 'zigzag', 'zspread', 'zeta']
        }
        
        # 新闻热词词典
        self.news_terms = {
            'a': ['ai', 'automation', 'antitrust', 'alliance', 'agreement', 'aid'],
            'b': ['blockchain', 'brexit', 'bailout', 'boycott', 'breakthrough', 'bipartisan'],
            'c': ['covid', 'climate', 'cyber', 'conflict', 'crisis', 'consolidation'],
            'd': ['digital', 'disruption', 'deal', 'deficit', 'diplomacy', 'downturn'],
            'e': ['election', 'epidemic', 'embargo', 'expansion', 'export', 'emergency'],
            'f': ['fintech', 'federal', 'friction', 'freeze', 'framework', 'fraud'],
            'g': ['globalization', 'green', 'game', 'guidance', 'glitch', 'glut'],
            'h': ['hack', 'housing', 'health', 'hostile', 'halt', 'hike'],
            'i': ['innovation', 'investigation', 'ipo', 'import', 'intervention', 'inquiry'],
            'j': ['joint', 'judge', 'journal', 'job', 'jubilee', 'jolt'],
            'k': ['key', 'korea', 'kremlin', 'kyoto', 'kickback', 'kneejerk'],
            'l': ['lawsuit', 'lockdown', 'labor', 'lobby', 'leak', 'landmark'],
            'm': ['merger', 'metaverse', 'monopoly', 'mandate', 'mission', 'milestone'],
            'n': ['negotiation', 'nuclear', 'national', 'network', 'neutrality', 'nudge'],
            'o': ['outbreak', 'oversight', 'outlook', 'opposition', 'occupation', 'offensive'],
            'p': ['pandemic', 'policy', 'probe', 'partnership', 'protest', 'pivot'],
            'q': ['quota', 'quarantine', 'question', 'qualm', 'quagmire', 'quake'],
            'r': ['regulation', 'reform', 'rally', 'recession', 'relief', 'restriction'],
            's': ['sanction', 'stimulus', 'supplychain', 'shortage', 'shutdown', 'settlement'],
            't': ['tariff', 'tech', 'tension', 'treaty', 'transition', 'turmoil'],
            'u': ['ukraine', 'unrest', 'upgrade', 'ultimatum', 'unwind', 'uplift'],
            'v': ['vaccine', 'virus', 'veto', 'venture', 'violation', 'volteface'],
            'w': ['war', 'wave', 'warning', 'waiver', 'workforce', 'woes'],
            'x': ['xenophobia', 'xray', 'xenolith', 'xerox', 'xmas'],
            'y': ['yield', 'yuan', 'youth', 'yearning', 'yammer', 'yank'],
            'z': ['zero', 'zone', 'zombie', 'zeal', 'zest', 'zigzag']
        }
        
        self.used_combinations = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        })
    
    def generate_random_letters(self) -> Tuple[str, str]:
        """随机生成两个不同的英文字母"""
        while True:
            letters = random.sample(string.ascii_lowercase, 2)
            combo = ''.join(sorted(letters))
            if combo not in self.used_combinations:
                self.used_combinations.add(combo)
                return tuple(letters)
    
    def search_terms(self, letter1: str, letter2: str, 
                     source: str = 'both') -> List[str]:
        """
        检索以两个字母拼音首字母开头的词汇
        
        Parameters:
            letter1, letter2: 随机生成的两个字母
            source: 'financial', 'news', 或 'both'
        
        Returns:
            候选词汇列表
        """
        candidates = []
        
        # 确保小写
        l1, l2 = letter1.lower(), letter2.lower()
        
        if source in ('financial', 'both'):
            # 获取两个字母开头的金融术语组合
            terms1 = self.financial_terms.get(l1, [])
            terms2 = self.financial_terms.get(l2, [])
            # 组合方式：两个词的组合，或单个词包含两个首字母
            for t1 in terms1:
                for t2 in terms2:
                    candidates.append(f"{t1}_{t2}")
                    candidates.append(f"{t1}{t2}")
        
        if source in ('news', 'both'):
            terms1 = self.news_terms.get(l1, [])
            terms2 = self.news_terms.get(l2, [])
            for t1 in terms1:
                for t2 in terms2:
                    candidates.append(f"{t1}_{t2}")
                    candidates.append(f"{t1}{t2}")
        
        # 去重并打乱
        candidates = list(set(candidates))
        random.shuffle(candidates)
        
        return candidates
    
    def web_search_enrichment(self, term: str, 
                              max_results: int = 5) -> List[str]:
        """
        通过网络搜索丰富词汇（简化版，实际可接入API）
        这里使用模拟数据演示框架
        """
        # 实际项目中可接入: Google Custom Search API, Bing API, 或百度API
        # 以下为模拟扩展
        extensions = {
            'momentum_earnings': ['earnings_momentum', 'momentum_adjusted_earnings', 
                                  'earnings_revision_momentum'],
            'volatility_value': ['volatility_adjusted_value', 'value_at_risk',
                                 'intrinsic_volatility'],
            'liquidity_growth': ['liquidity_adjusted_growth', 'growth_liquidity_ratio',
                                 'operational_liquidity_growth']
        }
        return extensions.get(term, [term + '_variant1', term + '_variant2'])


# ==================== 2. 因子计算与数据模拟模块 ====================

class FactorCalculator:
    """因子计算引擎 - 支持多种因子类型"""
    
    def __init__(self, price_data: pd.DataFrame, 
                 fundamental_data: Optional[pd.DataFrame] = None):
        """
        Parameters:
            price_data: 价格数据 DataFrame, index=date, columns=stocks
            fundamental_data: 基本面数据（可选）
        """
        self.price = price_data
        self.fundamental = fundamental_data
        self.returns = price_data.pct_change().dropna()
        self.stocks = price_data.columns.tolist()
        self.dates = price_data.index.tolist()
        
    def calculate_factor(self, factor_name: str, 
                         params: Optional[Dict] = None) -> pd.DataFrame:
        """
        根据因子名称计算因子值
        
        支持多种因子类型的映射计算
        """
        params = params or {}
        
        # 因子名称解析与映射
        factor_mapping = {
            # 动量类
            'momentum': self._momentum,
            'earnings_momentum': self._earnings_momentum,
            'price_momentum': self._price_momentum,
            'reversal': self._reversal,
            
            # 价值类
            'value': self._value,
            'bookvalue': self._book_to_market,
            'pe': self._pe_ratio,
            'pb': self._pb_ratio,
            
            # 波动率类
            'volatility': self._volatility,
            'beta': self._beta,
            'idiosyncratic_vol': self._idiosyncratic_vol,
            
            # 流动性类
            'liquidity': self._liquidity,
            'turnover': self._turnover,
            'amplitude': self._amplitude,
            
            # 质量类
            'quality': self._quality,
            'profit': self._profitability,
            'growth': self._growth,
            
            # 情绪类
            'sentiment': self._sentiment,
            'sentiment_momentum': self._sentiment_momentum,
            
            # 复合因子
            'momentum_earnings': self._momentum_earnings,
            'volatility_value': self._volatility_value,
            'liquidity_growth': self._liquidity_growth,
        }
        
        # 尝试匹配因子
        for key, func in factor_mapping.items():
            if key in factor_name.lower():
                return func(**params)
        
        # 默认：尝试从名称推断并生成随机但有结构的因子
        return self._generate_synthetic_factor(factor_name, **params)
    
    def _momentum(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """动量因子"""
        return self.price.pct_change(window).shift(1)  # 避免未来函数
    
    def _earnings_momentum(self, window: int = 60, **kwargs) -> pd.DataFrame:
        """盈利动量（模拟）"""
        if self.fundamental is not None and 'earnings' in self.fundamental.columns:
            return self.fundamental['earnings'].rolling(window).mean().shift(1)
        # 模拟：用价格动量代理
        return self._momentum(window) * (1 + np.random.normal(0, 0.1, self.price.shape))
    
    def _price_momentum(self, short: int = 20, long: int = 60, **kwargs) -> pd.DataFrame:
        """价格动量（短期/长期）"""
        short_mom = self._momentum(short)
        long_mom = self._momentum(long)
        return short_mom - long_mom
    
    def _reversal(self, window: int = 5, **kwargs) -> pd.DataFrame:
        """反转因子"""
        return -self._momentum(window)
    
    def _value(self, **kwargs) -> pd.DataFrame:
        """价值因子综合"""
        return self._book_to_market(**kwargs)
    
    def _book_to_market(self, **kwargs) -> pd.DataFrame:
        """账面市值比"""
        if self.fundamental is not None and 'book_value' in self.fundamental.columns:
            bv = self.fundamental['book_value']
            mv = self.price * self.fundamental.get('shares', 1)
            return (bv / mv).shift(1)
        # 模拟
        return pd.DataFrame(
            np.random.randn(*self.price.shape),
            index=self.price.index,
            columns=self.price.columns
        ).rolling(252).apply(lambda x: 1/x.iloc[-1] if x.iloc[-1] != 0 else 0).shift(1)
    
    def _pe_ratio(self, **kwargs) -> pd.DataFrame:
        """市盈率倒数（EP）"""
        # EP = E/P，越高越价值
        if self.fundamental is not None and 'earnings' in self.fundamental.columns:
            return (self.fundamental['earnings'] / self.price).shift(1)
        return pd.DataFrame(
            np.random.exponential(0.05, self.price.shape),
            index=self.price.index,
            columns=self.price.columns
        ).shift(1)
    
    def _pb_ratio(self, **kwargs) -> pd.DataFrame:
        """市净率倒数（BP）"""
        return self._book_to_market(**kwargs)
    
    def _volatility(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """波动率因子（低波动异常）"""
        return -self.returns.rolling(window).std().shift(1)
    
    def _beta(self, window: int = 60, **kwargs) -> pd.DataFrame:
        """Beta因子"""
        market = self.returns.mean(axis=1)  # 等权市场组合
        beta = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        for col in self.returns.columns:
            rolling_cov = self.returns[col].rolling(window).cov(market)
            rolling_var = market.rolling(window).var()
            beta[col] = rolling_cov / rolling_var
        return beta.shift(1)
    
    def _idiosyncratic_vol(self, window: int = 60, **kwargs) -> pd.DataFrame:
        """特质波动率"""
        market = self.returns.mean(axis=1)
        beta = self._beta(window)
        systematic = beta.multiply(market, axis=0)
        residual = self.returns - systematic
        return -residual.rolling(window).std().shift(1)
    
    def _liquidity(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """流动性因子"""
        return self._turnover(window, **kwargs)
    
    def _turnover(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """换手率因子（模拟）"""
        # 模拟换手率数据
        turnover = pd.DataFrame(
            np.random.lognormal(0, 1, self.price.shape),
            index=self.price.index,
            columns=self.price.columns
        )
        # 异常换手率（反转信号）
        turnover_ma = turnover.rolling(window).mean()
        return -(turnover / turnover_ma).shift(1)
    
    def _amplitude(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """振幅因子"""
        high = self.price * (1 + np.abs(np.random.randn(*self.price.shape)) * 0.02)
        low = self.price * (1 - np.abs(np.random.randn(*self.price.shape)) * 0.02)
        amplitude = (high - low) / self.price
        return -amplitude.rolling(window).mean().shift(1)
    
    def _quality(self, **kwargs) -> pd.DataFrame:
        """质量因子综合"""
        return self._profitability(**kwargs)
    
    def _profitability(self, **kwargs) -> pd.DataFrame:
        """盈利能力"""
        if self.fundamental is not None and 'roe' in self.fundamental.columns:
            return self.fundamental['roe'].shift(1)
        # 模拟ROE
        return pd.DataFrame(
            np.random.normal(0.1, 0.05, self.price.shape),
            index=self.price.index,
            columns=self.price.columns
        ).rolling(252).mean().shift(1)
    
    def _growth(self, window: int = 60, **kwargs) -> pd.DataFrame:
        """成长因子"""
        if self.fundamental is not None and 'revenue_growth' in self.fundamental.columns:
            return self.fundamental['revenue_growth'].shift(1)
        # 模拟营收增长
        return self.price.pct_change(window).shift(1) * 0.5 + \
               np.random.normal(0, 0.02, self.price.shape)
    
    def _sentiment(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """情绪因子"""
        # 模拟情绪指标：异常收益率的持续性
        abnormal = self.returns - self.returns.mean(axis=1).values.reshape(-1, 1)
        return abnormal.rolling(window).sum().shift(1)
    
    def _sentiment_momentum(self, window: int = 20, **kwargs) -> pd.DataFrame:
        """情绪动量"""
        sentiment = self._sentiment(window)
        momentum = self._momentum(window)
        return sentiment * momentum
    
    # 复合因子
    def _momentum_earnings(self, w_mom: float = 0.5, **kwargs) -> pd.DataFrame:
        """动量-盈利复合因子"""
        mom = self._momentum(**kwargs)
        earn = self._earnings_momentum(**kwargs)
        # 标准化后组合
        mom_norm = (mom - mom.mean()) / mom.std()
        earn_norm = (earn - earn.mean()) / earn.std()
        return w_mom * mom_norm + (1 - w_mom) * earn_norm
    
    def _volatility_value(self, w_vol: float = 0.5, **kwargs) -> pd.DataFrame:
        """波动率-价值复合因子"""
        vol = self._volatility(**kwargs)
        val = self._value(**kwargs)
        vol_norm = (vol - vol.mean()) / vol.std()
        val_norm = (val - val.mean()) / val.std()
        return w_vol * vol_norm + (1 - w_vol) * val_norm
    
    def _liquidity_growth(self, w_liq: float = 0.5, **kwargs) -> pd.DataFrame:
        """流动性-成长复合因子"""
        liq = self._liquidity(**kwargs)
        gr = self._growth(**kwargs)
        liq_norm = (liq - liq.mean()) / liq.std()
        gr_norm = (gr - gr.mean()) / gr.std()
        return w_liq * liq_norm + (1 - w_liq) * gr_norm
    
    def _generate_synthetic_factor(self, name: str, 
                                    seed: Optional[int] = None,
                                    persistence: float = 0.3,
                                    **kwargs) -> pd.DataFrame:
        """
        生成结构化合成因子（用于未知因子名称）
        
        生成具有指定自相关性和预测能力的合成因子
        """
        if seed is not None:
            np.random.seed(seed)
        
        n, m = self.price.shape
        
        # 生成具有持久性的因子（AR(1)过程）
        factor = pd.DataFrame(index=self.price.index, columns=self.price.columns)
        
        for col in self.price.columns:
            # 基础：随机游走成分
            random_walk = np.cumsum(np.random.randn(n) * 0.1)
            
            # 加入与未来收益相关的成分（模拟真实预测能力）
            future_returns = self.returns[col].shift(-1).fillna(0).values
            signal = np.convolve(future_returns, 
                                np.ones(20)/20, 
                                mode='same') * persistence
            
            # 组合
            combined = random_walk + signal * 10
            
            # 添加横截面相关性
            factor[col] = combined
        
        # 确保无未来函数
        return factor.shift(1).fillna(0)


# ==================== 3. IC 与 IR 计算模块 ====================

@dataclass
class ICMetrics:
    """IC分析指标"""
    ic_series: pd.Series  # IC时间序列
    ic_mean: float        # IC均值
    ic_std: float         # IC标准差
    ir: float             # 信息比率
    ic_win_rate: float    # IC胜率
    t_stat: float         # t统计量
    p_value: float        # p值
    ic_skew: float        # IC偏度
    ic_kurt: float        # IC峰度
    
    def is_valid(self, ir_threshold: float = 0.5, 
                 win_rate_threshold: float = 0.6,
                 max_ic_std: float = 0.5) -> bool:
        """
        判断因子是否有效
        
        标准：
        - IR > 0.5
        - IC胜率 > 60%
        - 不保留IC很高但波动极大的因子（IC标准差过大）
        """
        return (self.ir > ir_threshold and 
                self.ic_win_rate > win_rate_threshold and
                self.ic_std < max_ic_std)  # 剔除高波动因子


class ICAnalyzer:
    """IC（信息系数）分析器"""
    
    def __init__(self, forward_period: int = 1):
        """
        Parameters:
            forward_period: 预测期数（默认1期）
        """
        self.forward_period = forward_period
    
    def calculate_ic(self, factor_values: pd.DataFrame,
                     returns: pd.DataFrame,
                     method: str = 'spearman') -> pd.Series:
        """
        计算IC时间序列
        
        Parameters:
            factor_values: 因子值 DataFrame, index=date, columns=stocks
            returns: 未来收益 DataFrame, index=date, columns=stocks
            method: 'spearman' 或 'pearson'
        
        Returns:
            IC时间序列
        """
        # 对齐日期
        common_dates = factor_values.index.intersection(returns.index)
        factor = factor_values.loc[common_dates]
        ret = returns.loc[common_dates]
        
        ic_list = []
        ic_dates = []
        
        for date in common_dates:
            f = factor.loc[date].dropna()
            r = ret.loc[date].dropna()
            
            # 取交集
            common_stocks = f.index.intersection(r.index)
            if len(common_stocks) < 10:  # 最小样本量
                continue
            
            f_valid = f[common_stocks]
            r_valid = r[common_stocks]
            
            # 计算秩相关系数
            if method == 'spearman':
                corr, _ = stats.spearmanr(f_valid, r_valid)
            else:
                corr, _ = stats.pearsonr(f_valid, r_valid)
            
            if not np.isnan(corr):
                ic_list.append(corr)
                ic_dates.append(date)
        
        return pd.Series(ic_list, index=ic_dates, name='IC')
    
    def analyze(self, ic_series: pd.Series) -> ICMetrics:
        """
        全面分析IC序列
        
        IR = IC均值 / IC标准差
        IC胜率 = IC与均值同号的期数 / 总期数
        """
        ic_clean = ic_series.dropna()
        n = len(ic_clean)
        
        if n < 20:
            raise ValueError(f"IC样本量不足: {n} < 20")
        
        ic_mean = ic_clean.mean()
        ic_std = ic_clean.std()
        
        # IR计算
        ir = ic_mean / ic_std if ic_std > 0 else np.inf
        
        # IC胜率：与均值同号的比例
        same_sign = (ic_clean * ic_mean > 0).sum()
        ic_win_rate = same_sign / n
        
        # 统计检验
        t_stat, p_value = stats.ttest_1samp(ic_clean, 0)
        
        return ICMetrics(
            ic_series=ic_clean,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_win_rate=ic_win_rate,
            t_stat=t_stat,
            p_value=p_value / 2 if ic_mean > 0 else 1 - p_value / 2,  # 单侧p值
            ic_skew=ic_clean.skew(),
            ic_kurt=ic_clean.kurtosis()
        )
    
    def plot_ic(self, metrics: ICMetrics, title: str = "IC Analysis"):
        """可视化IC分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # IC时间序列
        ax1 = axes[0, 0]
        metrics.ic_series.plot(ax=ax1, color='steelblue', alpha=0.7)
        ax1.axhline(y=metrics.ic_mean, color='r', linestyle='--', label=f'Mean={metrics.ic_mean:.3f}')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.fill_between(metrics.ic_series.index, 0, metrics.ic_series, 
                        where=(metrics.ic_series > 0), alpha=0.3, color='green')
        ax1.fill_between(metrics.ic_series.index, 0, metrics.ic_series,
                        where=(metrics.ic_series < 0), alpha=0.3, color='red')
        ax1.set_title(f'IC Time Series (IR={metrics.ir:.2f}, WinRate={metrics.ic_win_rate:.1%})')
        ax1.legend()
        
        # IC分布
        ax2 = axes[0, 1]
        metrics.ic_series.hist(bins=30, ax=ax2, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=metrics.ic_mean, color='r', linestyle='--', linewidth=2)
        ax2.set_title(f'IC Distribution (Std={metrics.ic_std:.3f})')
        
        # IC累积
        ax3 = axes[1, 0]
        cum_ic = metrics.ic_series.cumsum()
        cum_ic.plot(ax=ax3, color='darkgreen')
        ax3.set_title('Cumulative IC')
        
        # 月度IC热力图（如果数据足够）
        ax4 = axes[1, 1]
        try:
            monthly_ic = metrics.ic_series.resample('M').mean()
            monthly_ic.index = monthly_ic.index.to_period('M')
            # 简化为柱状图
            monthly_ic.plot(kind='bar', ax=ax4, color='steelblue')
            ax4.set_title('Monthly IC')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        except:
            ax4.text(0.5, 0.5, 'Insufficient data for monthly view', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ==================== 4. 分层回测与单调性检验模块 ====================

@dataclass
class LayerBacktestResult:
    """分层回测结果"""
    layer_returns: pd.DataFrame      # 各层收益
    layer_cum_returns: pd.DataFrame  # 各层累积收益
    monotonicity_score: float        # 单调性得分
    is_monotonic: bool               # 是否单调
    top_minus_bottom: pd.Series      # 多空收益
    long_short_annual_return: float  # 多空年化收益
    long_short_max_drawdown: float   # 多空最大回撤
    long_short_sharpe: float         # 多空夏普比率
    
    # 换手率相关
    turnover_analysis: Dict          # 换手率分析结果


class LayerBacktester:
    """分层回测器"""
    
    def __init__(self, n_layers: int = 5, 
                 rebalance_freq: str = 'M',
                 holding_period: int = 20):
        """
        Parameters:
            n_layers: 分层数（默认5层）
            rebalance_freq: 调仓频率 'D', 'W', 'M'
            holding_period: 持有期（与调仓频率匹配）
        """
        self.n_layers = n_layers
        self.rebalance_freq = rebalance_freq
        self.holding_period = holding_period
    
    def _get_rebalance_dates(self, factor: pd.DataFrame) -> List[pd.Timestamp]:
        """获取调仓日期"""
        if self.rebalance_freq == 'D':
            return factor.index.tolist()
        elif self.rebalance_freq == 'W':
            return factor.resample('W-FRI').last().dropna().index.tolist()
        elif self.rebalance_freq == 'M':
            return factor.resample('M').last().dropna().index.tolist()
        else:
            raise ValueError(f"Unsupported frequency: {self.rebalance_freq}")
    
    def backtest(self, factor: pd.DataFrame,
                 returns: pd.DataFrame,
                 price: pd.DataFrame) -> LayerBacktestResult:
        """
        执行分层回测
        
        Parameters:
            factor: 因子值
            returns: 未来收益
            price: 价格数据（用于计算换手率）
        
        Returns:
            分层回测结果
        """
        rebalance_dates = self._get_rebalance_dates(factor)
        
        # 存储各层收益
        layer_returns_list = []
        turnover_records = []
        
        prev_holdings = None
        
        for i, date in enumerate(rebalance_dates[:-1]):
            if date not in factor.index or date not in returns.index:
                continue
            
            # 获取当前因子值并排序分层
            f = factor.loc[date].dropna()
            if len(f) < self.n_layers * 5:  # 每层至少5只股票
                continue
            
            # 分层
            f_ranked = f.rank(pct=True)
            layer_labels = pd.qcut(f_ranked, self.n_layers, 
                                  labels=range(1, self.n_layers + 1),
                                  duplicates='drop')
            
            # 计算各层下一期收益
            next_date = rebalance_dates[i + 1]
            if next_date not in returns.index:
                # 找下一个有效日期
                future_dates = returns.index[returns.index > date]
                if len(future_dates) == 0:
                    continue
                next_date = future_dates[0]
            
            period_returns = returns.loc[next_date]
            
            layer_ret = {}
            current_holdings = {}
            
            for layer in range(1, self.n_layers + 1):
                stocks_in_layer = layer_labels[layer_labels == layer].index
                # 等权收益
                valid_stocks = stocks_in_layer.intersection(period_returns.index)
                if len(valid_stocks) > 0:
                    avg_ret = period_returns[valid_stocks].mean()
                    layer_ret[f'L{layer}'] = avg_ret
                    current_holdings[f'L{layer}'] = set(valid_stocks)
                else:
                    layer_ret[f'L{layer}'] = 0
            
            layer_returns_list.append(pd.Series(layer_ret, name=next_date))
            
            # 计算换手率
            if prev_holdings is not None:
                turnover = self._calculate_turnover(
                    prev_holdings, current_holdings
                )
                turnover_records.append(turnover)
            
            prev_holdings = current_holdings
        
        # 整理结果
        layer_returns_df = pd.DataFrame(layer_returns_list)
        
        # 单调性检验
        monotonicity = self._test_monotonicity(layer_returns_df)
        
        # 多空组合（最高层 - 最低层）
        top_col = f'L{self.n_layers}'
        bottom_col = 'L1'
        long_short = layer_returns_df[top_col] - layer_returns_df[bottom_col]
        
        # 多空绩效
        ls_cum = (1 + long_short).cumprod()
        ls_total_ret = ls_cum.iloc[-1] - 1
        
        # 年化收益
        n_periods = len(long_short)
        periods_per_year = self._periods_per_year()
        ls_annual = (1 + ls_total_ret) ** (periods_per_year / n_periods) - 1
        
        # 最大回撤
        ls_max_dd = self._max_drawdown(ls_cum)
        
        # 夏普
        ls_sharpe = long_short.mean() / long_short.std() * np.sqrt(periods_per_year)
        
        # 换手率分析
        turnover_analysis = self._analyze_turnover(turnover_records)
        
        # 累积收益
        layer_cum = (1 + layer_returns_df).cumprod()
        
        return LayerBacktestResult(
            layer_returns=layer_returns_df,
            layer_cum_returns=layer_cum,
            monotonicity_score=monotonicity['score'],
            is_monotonic=monotonicity['is_monotonic'],
            top_minus_bottom=long_short,
            long_short_annual_return=ls_annual,
            long_short_max_drawdown=ls_max_dd,
            long_short_sharpe=ls_sharpe,
            turnover_analysis=turnover_analysis
        )
    
    def _periods_per_year(self) -> float:
        """每年期数"""
        mapping = {'D': 252, 'W': 52, 'M': 12}
        return mapping.get(self.rebalance_freq, 252)
    
    def _calculate_turnover(self, prev: Dict, curr: Dict) -> Dict:
        """计算换手率"""
        turnover = {}
        for layer in prev.keys():
            if layer in curr:
                prev_set = prev[layer]
                curr_set = curr[layer]
                if len(prev_set) > 0:
                    # 换手率 = 1 - 交集/并集 的补，或简单：卖出+买入/2
                    intersection = len(prev_set & curr_set)
                    turnover[layer] = 1 - intersection / len(prev_set)
                else:
                    turnover[layer] = 0
            else:
                turnover[layer] = 1
        return turnover
    
    def _analyze_turnover(self, turnover_records: List[Dict]) -> Dict:
        """分析换手率是否过高"""
        if not turnover_records:
            return {'avg_turnover': 0, 'is_excessive': False, 'details': {}}
        
        df = pd.DataFrame(turnover_records)
        avg_turnover = df.mean().mean()
        
        # 判断标准：月换手率>50%为过高（假设月频）
        # 不同频率调整
        threshold = {'D': 0.3, 'W': 0.4, 'M': 0.5}.get(self.rebalance_freq, 0.5)
        
        # 投资周期匹配：持有期与调仓频率
        expected_holding = self.holding_period
        # 如果换手导致平均持有期远短于预期，则不匹配
        
        return {
            'avg_turnover': avg_turnover,
            'is_excessive': avg_turnover > threshold,
            'details': df.mean().to_dict(),
            'threshold': threshold,
            'holding_period_match': avg_turnover < 1 / (expected_holding * 0.5)
        }
    
    def _test_monotonicity(self, layer_returns: pd.DataFrame) -> Dict:
        """
        检验分层收益的单调性
        
        理想情况：L1 < L2 < L3 < L4 < L5（因子方向正确）
        或 L1 > L2 > L3 > L4 > L5（反向因子）
        """
        if len(layer_returns) < 10:
            return {'score': 0, 'is_monotonic': False}
        
        # 计算各层平均收益
        mean_returns = layer_returns.mean()
        
        # 斯皮尔曼秩相关检验
        ranks = np.arange(1, self.n_layers + 1)
        corr, pvalue = stats.spearmanr(ranks, mean_returns.values)
        
        # 单调性得分：相关系数的绝对值，且p值显著
        score = abs(corr) if pvalue < 0.1 else 0
        
        # 严格单调：各层顺序正确且显著
        is_monotonic = (score > 0.6) and (pvalue < 0.05)
        
        # 额外检验：相邻层差异的一致性
        diffs = np.diff(mean_returns.values)
        consistency = (diffs > 0).all() or (diffs < 0).all()
        
        return {
            'score': score,
            'is_monotonic': is_monotonic and consistency,
            'direction': 'positive' if corr > 0 else 'negative',
            'pvalue': pvalue,
            'mean_returns': mean_returns.to_dict(),
            'consistency': consistency
        }
    
    def plot_layers(self, result: LayerBacktestResult, title: str = "Layer Backtest"):
        """可视化分层回测"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 各层累积收益
        ax1 = axes[0, 0]
        for col in result.layer_cum_returns.columns:
            result.layer_cum_returns[col].plot(ax=ax1, label=col, linewidth=2)
        ax1.set_title('Cumulative Returns by Layer')
        ax1.legend()
        ax1.set_ylabel('Cumulative Return')
        
        # 各层收益分布
        ax2 = axes[0, 1]
        result.layer_returns.boxplot(ax=ax2)
        ax2.set_title('Return Distribution by Layer')
        
        # 多空收益
        ax3 = axes[1, 0]
        ls_cum = (1 + result.top_minus_bottom).cumprod()
        ls_cum.plot(ax=ax3, color='purple', linewidth=2)
        ax3.fill_between(ls_cum.index, 1, ls_cum, 
                        where=(ls_cum > 1), alpha=0.3, color='green')
        ax3.fill_between(ls_cum.index, 1, ls_cum,
                        where=(ls_cum < 1), alpha=0.3, color='red')
        ax3.set_title(f'Long-Short Cumulative (AnnRet={result.long_short_annual_return:.1%}, '
                     f'MDD={result.long_short_max_drawdown:.1%})')
        ax3.axhline(y=1, color='k', linestyle='-', alpha=0.3)
        
        # 单调性可视化
        ax4 = axes[1, 1]
        means = pd.Series({k: v for k, v in result.layer_returns.mean().items()})
        colors = ['red' if x < 0 else 'green' for x in means.values]
        means.plot(kind='bar', ax=ax4, color=colors, edgecolor='black')
        ax4.set_title(f'Monotonicity: {"PASS" if result.is_monotonic else "FAIL"} '
                     f'(Score={result.monotonicity_score:.2f})')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ==================== 5. 主控流程：因子挖掘引擎 ====================

@dataclass
class FactorMiningResult:
    """因子挖掘最终结果"""
    factor_name: str
    letters: Tuple[str, str]
    source_term: str
    ic_metrics: Optional[ICMetrics]
    layer_result: Optional[LayerBacktestResult]
    is_valid: bool
    failure_reason: Optional[str]
    validation_stage: str  # 'ic', 'layer', 'long_short', 'turnover', 'pass'


class FactorMiningEngine:
    """
    金融因子挖掘主引擎
    
    完整流程：
    1. 随机生成两个字母
    2. 检索对应词汇
    3. 计算因子并IC/IR检验
    4. 分层回测检验单调性
    5. 多空收益与最大回撤检验
    6. 换手率与投资周期匹配检验
    """
    
    def __init__(self, 
                 price_data: pd.DataFrame,
                 fundamental_data: Optional[pd.DataFrame] = None,
                 max_attempts: int = 100,
                 random_seed: Optional[int] = None):
        """
        Parameters:
            price_data: 股票价格数据
            fundamental_data: 基本面数据（可选）
            max_attempts: 最大尝试次数
        """
        self.price = price_data
        self.fundamental = fundamental_data
        self.max_attempts = max_attempts
        
        # 初始化组件
        self.vocab_gen = VocabularyGenerator()
        self.factor_calc = FactorCalculator(price_data, fundamental_data)
        self.ic_analyzer = ICAnalyzer(forward_period=1)
        self.layer_backtester = LayerBacktester(
            n_layers=5, 
            rebalance_freq='M',
            holding_period=20
        )
        
        # 参数配置
        self.ir_threshold = 0.5
        self.ic_win_rate_threshold = 0.6
        self.max_ic_std = 0.5  # 剔除高波动IC
        self.min_ls_annual_return = 0.05  # 多空年化最低5%
        self.max_ls_drawdown = 0.10  # 最大回撤10%
        self.max_turnover_threshold = 0.5  # 最高换手率
        
        # 结果记录
        self.results: List[FactorMiningResult] = []
        self.valid_factors: List[FactorMiningResult] = []
        
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def run(self, verbose: bool = True) -> List[FactorMiningResult]:
        """
        运行完整的因子挖掘流程
        
        Returns:
            所有验证通过的因子列表
        """
        print("=" * 80)
        print("启动金融因子挖掘引擎")
        print(f"数据区间: {self.price.index[0]} ~ {self.price.index[-1]}")
        print(f"股票数量: {len(self.price.columns)}")
        print("=" * 80)
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n{'='*40}")
            print(f"尝试 #{attempt}")
            print(f"{'='*40}")
            
            result = self._single_attempt()
            self.results.append(result)
            
            if result.is_valid:
                self.valid_factors.append(result)
                print(f"\n✅ 发现有效因子: {result.factor_name}")
                print(f"   IR={result.ic_metrics.ir:.3f}, "
                      f"IC胜率={result.ic_metrics.ic_win_rate:.1%}, "
                      f"多空收益={result.layer_result.long_short_annual_return:.1%}")
            
            if len(self.valid_factors) >= 5:  # 找到5个有效因子后停止
                print(f"\n已找到 {len(self.valid_factors)} 个有效因子，停止搜索")
                break
            
            time.sleep(0.1)  # 避免过快
        
        print(f"\n{'='*80}")
        print(f"挖掘完成: 尝试{attempt}次, 发现{len(self.valid_factors)}个有效因子")
        print(f"{'='*80}")
        
        return self.valid_factors
    
    def _single_attempt(self) -> FactorMiningResult:
        """单次因子挖掘尝试"""
        
        # Step 1: 生成随机字母
        letter1, letter2 = self.vocab_gen.generate_random_letters()
        print(f"随机字母: [{letter1.upper()}, {letter2.upper()}]")
        
        # Step 2: 检索词汇
        terms = self.vocab_gen.search_terms(letter1, letter2, source='both')
        print(f"候选词汇数: {len(terms)}")
        
        # 尝试每个候选词
        for term in terms[:10]:  # 最多试前10个
            print(f"\n  测试因子: {term}")
            
            # Step 3: 计算因子值
            try:
                factor_values = self.factor_calc.calculate_factor(term)
            except Exception as e:
                print(f"    因子计算失败: {e}")
                continue
            
            # 检查数据有效性
            if factor_values.isna().all().all():
                continue
            
            # Step 4: IC/IR检验
            try:
                # 准备未来收益
                future_returns = self.price.pct_change().shift(-1).dropna()
                
                ic_series = self.ic_analyzer.calculate_ic(
                    factor_values, future_returns, method='spearman'
                )
                
                if len(ic_series) < 20:
                    print(f"    IC样本不足")
                    continue
                
                ic_metrics = self.ic_analyzer.analyze(ic_series)
                
                print(f"    IC均值={ic_metrics.ic_mean:.3f}, "
                      f"IC标准差={ic_metrics.ic_std:.3f}, "
                      f"IR={ic_metrics.ir:.3f}, "
                      f"胜率={ic_metrics.ic_win_rate:.1%}")
                
                # IR > 0.5, 胜率 > 60%, 且IC波动不过大
                if not ic_metrics.is_valid(self.ir_threshold, 
                                           self.ic_win_rate_threshold,
                                           self.max_ic_std):
                    fail_reason = []
                    if ic_metrics.ir <= self.ir_threshold:
                        fail_reason.append(f"IR={ic_metrics.ir:.3f}<={self.ir_threshold}")
                    if ic_metrics.ic_win_rate <= self.ic_win_rate_threshold:
                        fail_reason.append(f"胜率={ic_metrics.ic_win_rate:.1%}<={self.ic_win_rate_threshold:.1%}")
                    if ic_metrics.ic_std >= self.max_ic_std:
                        fail_reason.append(f"IC波动={ic_metrics.ic_std:.3f}>={self.max_ic_std}")
                    
                    print(f"    ❌ IC检验失败: {', '.join(fail_reason)}")
                    return FactorMiningResult(
                        factor_name=term,
                        letters=(letter1, letter2),
                        source_term=term,
                        ic_metrics=ic_metrics,
                        layer_result=None,
                        is_valid=False,
                        failure_reason=f"IC检验失败: {', '.join(fail_reason)}",
                        validation_stage='ic'
                    )
                
                print(f"    ✅ IC检验通过")
                
            except Exception as e:
                print(f"    IC计算异常: {e}")
                continue
            
            # Step 5: 分层回测 - 检验单调性
            try:
                layer_result = self.layer_backtester.backtest(
                    factor_values, 
                    self.price.pct_change().shift(-1).dropna(),
                    self.price
                )
                
                print(f"    单调性得分={layer_result.monotonicity_score:.3f}, "
                      f"单调={layer_result.is_monotonic}")
                
                if not layer_result.is_monotonic:
                    print(f"    ❌ 单调性检验失败")
                    return FactorMiningResult(
                        factor_name=term,
                        letters=(letter1, letter2),
                        source_term=term,
                        ic_metrics=ic_metrics,
                        layer_result=layer_result,
                        is_valid=False,
                        failure_reason="分层收益非单调",
                        validation_stage='layer'
                    )
                
                print(f"    ✅ 单调性检验通过")
                
            except Exception as e:
                print(f"    分层回测异常: {e}")
                continue
            
            # Step 6: 多空收益与最大回撤检验
            ls_annual = layer_result.long_short_annual_return
            ls_mdd = layer_result.long_short_max_drawdown
            
            print(f"    多空年化={ls_annual:.1%}, 最大回撤={ls_mdd:.1%}")
            
            if ls_annual <= self.min_ls_annual_return:
                print(f"    ❌ 多空收益不足")
                return FactorMiningResult(
                    factor_name=term,
                    letters=(letter1, letter2),
                    source_term=term,
                    ic_metrics=ic_metrics,
                    layer_result=layer_result,
                    is_valid=False,
                    failure_reason=f"多空年化收益{ls_annual:.1%}<={self.min_ls_annual_return:.1%}",
                    validation_stage='long_short'
                )
            
            if ls_mdd >= self.max_ls_drawdown:
                print(f"    ❌ 最大回撤过大")
                return FactorMiningResult(
                    factor_name=term,
                    letters=(letter1, letter2),
                    source_term=term,
                    ic_metrics=ic_metrics,
                    layer_result=layer_result,
                    is_valid=False,
                    failure_reason=f"最大回撤{ls_mdd:.1%}>={self.max_ls_drawdown:.1%}",
                    validation_stage='long_short'
                )
            
            print(f"    ✅ 多空收益/回撤检验通过")
            
            # Step 7: 换手率与投资周期匹配检验
            turnover = layer_result.turnover_analysis
            
            print(f"    平均换手率={turnover['avg_turnover']:.1%}, "
                  f"阈值={turnover['threshold']:.1%}")
            
            if turnover['is_excessive'] or not turnover.get('holding_period_match', True):
                print(f"    ❌ 换手率过高或与投资周期不匹配")
                return FactorMiningResult(
                    factor_name=term,
                    letters=(letter1, letter2),
                    source_term=term,
                    ic_metrics=ic_metrics,
                    layer_result=layer_result,
                    is_valid=False,
                    failure_reason=f"换手率{turnover['avg_turnover']:.1%}过高或不匹配",
                    validation_stage='turnover'
                )
            
            print(f"    ✅ 换手率检验通过")
            
            # 全部通过！
            return FactorMiningResult(
                factor_name=term,
                letters=(letter1, letter2),
                source_term=term,
                ic_metrics=ic_metrics,
                layer_result=layer_result,
                is_valid=True,
                failure_reason=None,
                validation_stage='pass'
            )
        
        # 所有候选词都失败
        return FactorMiningResult(
            factor_name="N/A",
            letters=(letter1, letter2),
            source_term="N/A",
            ic_metrics=None,
            layer_result=None,
            is_valid=False,
            failure_reason="无有效候选因子",
            validation_stage='vocabulary'
        )
    
    def generate_report(self, save_path: Optional[str] = None):
        """生成完整报告"""
        report = []
        report.append("=" * 80)
        report.append("因子挖掘完整报告")
        report.append("=" * 80)
        report.append(f"总尝试次数: {len(self.results)}")
        report.append(f"有效因子数: {len(self.valid_factors)}")
        report.append(f"成功率: {len(self.valid_factors)/len(self.results)*100:.1f}%")
        report.append("")
        
        # 失败原因统计
        failure_reasons = defaultdict(int)
        for r in self.results:
            if not r.is_valid:
                failure_reasons[r.validation_stage] += 1
        
        report.append("失败阶段分布:")
        for stage, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            report.append(f"  {stage}: {count}次")
        report.append("")
        
        # 有效因子详情
        if self.valid_factors:
            report.append("-" * 80)
            report.append("有效因子详情:")
            report.append("-" * 80)
            
            for i, vf in enumerate(self.valid_factors, 1):
                report.append(f"\n{i}. {vf.factor_name}")
                report.append(f"   来源字母: {vf.letters[0].upper()}-{vf.letters[1].upper()}")
                report.append(f"   IC均值: {vf.ic_metrics.ic_mean:.4f}")
                report.append(f"   IR: {vf.ic_metrics.ir:.3f}")
                report.append(f"   IC胜率: {vf.ic_metrics.ic_win_rate:.1%}")
                report.append(f"   多空年化: {vf.layer_result.long_short_annual_return:.2%}")
                report.append(f"   多空最大回撤: {vf.layer_result.long_short_max_drawdown:.2%}")
                report.append(f"   单调性: {'是' if vf.layer_result.is_monotonic else '否'}")
        
        report_text = '\n'.join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text


# ==================== 6. 模拟数据生成与演示 ====================

def generate_simulated_data(n_stocks: int = 100,
                            n_days: int = 500,
                            start_date: str = '2020-01-01') -> pd.DataFrame:
    """
    生成模拟股票数据，包含一些可预测结构
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    
    # 生成具有因子结构的价格数据
    # 基础：随机游走 + 共同因子 + 特质因子
    
    # 共同市场因子
    market = np.cumsum(np.random.randn(n_days) * 0.01)
    
    # 股票特定参数
    betas = np.random.uniform(0.5, 1.5, n_stocks)
    vols = np.random.uniform(0.01, 0.03, n_stocks)
    
    # 生成价格
    prices = pd.DataFrame(index=dates, columns=[f'S{i:04d}' for i in range(n_stocks)])
    
    for i in range(n_stocks):
        # 共同成分
        common = betas[i] * market
        # 特质成分（包含一些可预测模式）
        idio = np.cumsum(np.random.randn(n_days) * vols[i])
        
        # 加入动量效应（短期可预测）
        momentum = np.convolve(idio, np.ones(5)/5, mode='same') * 0.1
        
        # 加入价值效应（长期均值回归）
        value = -0.05 * (common - common.mean())
        
        log_price = common + idio + momentum + value
        prices.iloc[:, i] = np.exp(log_price) * 10  # 基准价格10元
    
    return prices


def main():
    """主程序演示"""
    
    print("生成模拟数据...")
    price_data = generate_simulated_data(n_stocks=100, n_days=500)
    print(f"数据形状: {price_data.shape}")
    print(f"日期范围: {price_data.index[0]} ~ {price_data.index[-1]}")
    
    # 创建并运行挖掘引擎
    engine = FactorMiningEngine(
        price_data=price_data,
        max_attempts=50,
        random_seed=2024
    )
    
    # 运行挖掘
    valid_factors = engine.run(verbose=True)
    
    # 生成报告
    engine.generate_report(save_path='factor_mining_report.txt')
    
    # 可视化最佳因子
    if valid_factors:
        best = valid_factors[0]
        print(f"\n可视化最佳因子: {best.factor_name}")
        
        # IC图
        fig_ic = engine.ic_analyzer.plot_ic(
            best.ic_metrics, 
            title=f"IC Analysis: {best.factor_name}"
        )
        plt.savefig(f'ic_{best.factor_name}.png', dpi=150, bbox_inches='tight')
        
        # 分层回测图
        fig_layer = engine.layer_backtester.plot_layers(
            best.layer_result,
            title=f"Layer Backtest: {best.factor_name}"
        )
        plt.savefig(f'layer_{best.factor_name}.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    return engine, valid_factors


if __name__ == "__main__":
    engine, factors = main()
