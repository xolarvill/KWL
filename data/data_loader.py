# data_loader.py
class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_individual_data(self, path):
        """加载个体面板数据(CFPS)"""
        # 使用pandas读取dta文件
        # 返回处理后的数据框
        
    def load_regional_data(self, path):
        """加载地区特征数据"""
        # 使用pandas读取xlsx文件
        # 返回处理后的数据框
        
    def load_adjacency_matrix(self, path):
        """加载地区临近矩阵"""
        # 加载并处理临近矩阵
        # 返回处理后的矩阵
        
    def merge_data(self):
        """合并不同来源的数据"""
        # 根据地区和时间合并数据
        # 返回合并后的数据集