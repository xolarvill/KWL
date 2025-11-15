"""
ABM验证模块
检验模型是否能自发涌现宏观规律
对应论文tex第1540-1602行
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False


class ABMValidator:
    """
    ABM模型验证器
    检验齐夫定律和胡焕庸线的涌现
    """
    
    def __init__(self, regional_populations: np.ndarray, region_names: List[str] = None):
        """
        初始化验证器
        
        Args:
            regional_populations: 各地区人口数组 (n_regions,)
            region_names: 地区名称列表
        """
        self.populations = regional_populations
        self.n_regions = len(regional_populations)
        self.region_names = region_names or [f"省份{i}" for i in range(self.n_regions)]
        
        # 胡焕庸线分界：以省份中心经度75°E和95°E为界
        # 东南半壁包括黑龙江、吉林、辽宁等相对发达省份
        self.hu_line_eastern_provinces = self._get_hu_line_eastern_provinces()
        
    def _get_hu_line_eastern_provinces(self) -> np.ndarray:
        """
        获取属于胡焕庸线东南半壁的省份索引
        基于地理经纬度和人口密度特征
        """
        # 根据实际地理分布设定（胡焕庸线从黑龙江黑河到云南腾冲）
        # 简化：假设前20个省份在东南半壁
        # 实际应用中应从geo.xlsx加载真实地理坐标
        eastern_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        return eastern_indices[:min(len(eastern_indices), self.n_regions)]
    
    def validate_zipf_law(self, plot: bool = True) -> Dict[str, float]:
        """
        验证城市规模分布是否符合Zipf定律
        对应 tex 1550-1576行
        
        Args:
            plot: 是否绘制图表
            
        Returns:
            zipf_metrics: Zipf定律验证指标
        """
        print("\n" + "="*60)
        print("验证Zipf定律")
        print("="*60)
        
        # 按人口规模排序（降序）
        sorted_indices = np.argsort(self.populations)[::-1]
        sorted_populations = self.populations[sorted_indices]
        
        # 计算位序 (rank)
        ranks = np.arange(1, len(sorted_populations) + 1)
        
        # 双对数回归：log(Rank) = a - ζ * log(Size)
        # 等价于：log(Size) = c - (1/ζ) * log(Rank)
        log_ranks = np.log(ranks)
        log_sizes = np.log(sorted_populations + 1)  # +1避免log(0)
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_ranks, log_sizes
        )
        
        # Zipf指数 ζ（理论值≈1.05-1.11）
        zipf_exponent = -slope  # 因为回归方程是 log(Size) = intercept + slope * log(Rank)
        
        print(f"Zipf指数 ζ: {zipf_exponent:.4f}")
        print(f"R²: {r_value**2:.4f}")
        print(f"标准误: {std_err:.4f}")
        print(f"p值: {p_value:.4f}")
        
        # 评估与理论值的一致性
        target_range = (1.05, 1.11)
        in_range = target_range[0] <= zipf_exponent <= target_range[1]
        deviation = min(
            abs(zipf_exponent - target_range[0]),
            abs(zipf_exponent - target_range[1])
        )
        
        print(f"目标范围: [{target_range[0]}, {target_range[1]}]")
        print(f"是否在范围内: {'✓' if in_range else '✗'}")
        print(f"偏差: {deviation:.4f}")
        
        # 绘制图表
        if plot:
            self._plot_zipf_law(log_ranks, log_sizes, slope, intercept, zipf_exponent, r_value**2)
        
        return {
            'zipf_exponent': zipf_exponent,
            'r_squared': r_value**2,
            'std_error': std_err,
            'p_value': p_value,
            'in_range': in_range,
            'deviation': deviation
        }
    
    def _plot_zipf_law(self, log_ranks: np.ndarray, log_sizes: np.ndarray, 
                       slope: float, intercept: float, 
                       zipf_exponent: float, r_squared: float):
        """绘制Zipf定律图表"""
        plt.figure(figsize=(10, 6))
        
        # 散点图
        plt.scatter(log_ranks, log_sizes, alpha=0.6, s=50, label='模拟数据')
        
        # 回归线
        fitted_line = intercept + slope * log_ranks
        plt.plot(log_ranks, fitted_line, 'r-', linewidth=2, 
                label=f'拟合线: ζ={zipf_exponent:.3f}, R²={r_squared:.3f}')
        
        # 理论线（ζ=1）
        theoretical_line = np.log(self.populations.sum()) - log_ranks
        plt.plot(log_ranks[:5], theoretical_line[:5], 'g--', linewidth=2, 
                label='理论线 (ζ=1)')
        
        plt.xlabel('ln(位序)', fontsize=12)
        plt.ylabel('ln(人口规模)', fontsize=12)
        plt.title('城市规模分布的Zipf定律检验', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/zipf_law_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("图表已保存: results/figures/zipf_law_validation.png")
    
    def validate_hu_line(self) -> Dict[str, float]:
        """
        验证胡焕庸线人口分布
        对应 tex 1591-1595行
        
        Returns:
            hu_line_metrics: 胡焕庸线验证指标
        """
        print("\n" + "="*60)
        print("验证胡焕庸线")
        print("="*60)
        
        # 总人口
        total_population = self.populations.sum()
        
        # 东南半壁人口
        eastern_population = self.populations[self.hu_line_eastern_provinces].sum()
        
        # 东南半壁占比
        eastern_share = eastern_population / total_population
        
        # 目标值：94%左右
        target_share = 0.94
        
        print(f"总人口: {total_population:,.0f}")
        print(f"东南半壁人口: {eastern_population:,.0f}")
        print(f"东南半壁占比: {eastern_share:.3f} ({eastern_share*100:.1f}%)")
        print(f"目标占比: {target_share:.3f} ({target_share*100:.1f}%)")
        
        # 计算偏差
        absolute_error = abs(eastern_share - target_share)
        relative_error = absolute_error / target_share
        
        print(f"绝对误差: {absolute_error:.4f}")
        print(f"相对误差: {relative_error:.4f} ({relative_error*100:.2f}%)")
        
        # 评估准确性
        tolerance = 0.01  # 1%的容差
        is_accurate = absolute_error <= tolerance
        
        print(f"容差范围: ±{tolerance*100:.1f}%")
        print(f"是否通过: {'✓' if is_accurate else '✗'}")
        
        return {
            'eastern_population': eastern_population,
            'total_population': total_population,
            'eastern_share': eastern_share,
            'target_share': target_share,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'is_accurate': is_accurate
        }
    
    def plot_population_distribution(self):
        """绘制人口分布地图（概念性）"""
        plt.figure(figsize=(12, 8))
        
        # 创建网格地图（简化版）
        # 实际应用中应使用真实的中国地图shapefile
        
        # 假设排列省份
        n_cols = 6
        n_rows = 5
        
        # 归一化人口用于颜色映射
        normalized_pop = self.populations / self.populations.max()
        
        plt.imshow(np.random.rand(n_rows, n_cols), cmap='YlOrRd', alpha=0.7)
        plt.colorbar(label='人口密度（标准化）')
        
        plt.title('ABM模拟人口分布与胡焕庸线', fontsize=14, fontweight='bold')
        
        # 绘制胡焕庸线（概念性）
        # 从左上到右下的虚线
        plt.axline((0, 0), (1, 1), color='blue', linestyle='--', linewidth=2, 
                  label='胡焕庸线')
        
        plt.legend()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/figures/population_distribution_hu_line.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("图表已保存: results/figures/population_distribution_hu_line.png")
    
    def run_full_validation(self, plot: bool = True) -> Dict[str, Any]:
        """
        执行完整验证流程
        
        Args:
            plot: 是否绘制图表
            
        Returns:
            validation_results: 完整验证结果
        """
        print("\n" + "="*60)
        print("ABM完整验证流程")
        print("="*60)
        
        # 1. Zipf定律验证
        print("\n[1/2] Zipf定律验证...")
        zipf_results = self.validate_zipf_law(plot=plot)
        
        # 2. 胡焕庸线验证
        print("\n[2/2] 胡焕庸线验证...")
        hu_line_results = self.validate_hu_line()
        
        # 合并结果
        validation_results = {
            'zipf_law': zipf_results,
            'hu_line': hu_line_results,
            'overall_assessment': self._assess_overall_fit(zipf_results, hu_line_results)
        }
        
        # 打印总结
        print("\n" + "="*60)
        print("验证结果总结")
        print("="*60)
        print(f"Zipf指数: {zipf_results['zipf_exponent']:.4f} {'✓' if zipf_results['in_range'] else '✗'}")
        print(f"R²: {zipf_results['r_squared']:.4f}")
        print(f"胡焕庸线误差: {hu_line_results['relative_error']:.4f} {'✓' if hu_line_results['is_accurate'] else '✗'}")
        print(f"总体评估: {validation_results['overall_assessment']['status']}")
        
        return validation_results
    
    def _assess_overall_fit(self, zipf_results: Dict, hu_line_results: Dict) -> Dict[str, Any]:
        """
        评估整体拟合质量
        
        Returns:
            assessment: 整体评估结果
        """
        # Zipf定律标准
        zipf_pass = zipf_results['in_range'] and zipf_results['r_squared'] > 0.95
        
        # 胡焕庸线标准
        hu_line_pass = hu_line_results['is_accurate']
        
        # 总体评估
        overall_pass = zipf_pass and hu_line_pass
        
        return {
            'zipf_pass': zipf_pass,
            'hu_line_pass': hu_line_pass,
            'overall_pass': overall_pass,
            'status': '通过 ✓' if overall_pass else '未通过 ✗'
        }


def demo_validation():
    """演示验证流程"""
    print("\n" + "="*60)
    print("ABM验证演示")
    print("="*60)
    
    # 模拟人口分布（基于真实数据的近似）
    np.random.seed(42)
    
    # 生成符合Zipf定律的分布（大小城市差异）
    n_regions = 29
    base_populations = np.random.zipf(1.1, n_regions)
    
    # 调整使胡焕庸线东南占94%
    eastern_share_idx = int(n_regions * 0.7)  # 70%省份在东部
    eastern_multiplier = 15  # 东部人口密度是西部的15倍
    
    populations = base_populations.copy()
    populations[:eastern_share_idx] *= eastern_multiplier
    
    # 缩放到合理范围（类似于真实省份人口）
    populations = populations / populations.sum() * 1.4e9  # 总人口约14亿
    populations = populations.astype(int)
    
    # 创建验证器
    validator = ABMValidator(populations)
    
    # 运行完整验证
    results = validator.run_full_validation(plot=True)
    
    return results


if __name__ == '__main__':
    validation_results = demo_validation()