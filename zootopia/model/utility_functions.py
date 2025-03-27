from config import ModelConfig
from migration_parameters import MigrationParameters

class UtilityFunctions:
    def __init__(self, config: ModelConfig, params: MigrationParameters):
        self.config = config
        self.params = params # param.gamma1读取后直接为torch.tensor类型
        self.beta = self.config.discount_factor
    
    def utility_function(self, state, tau_type):
        """计算效用函数"""
        # 计算给定状态和tau类型的效用函数
        # 这里需要根据具体的模型和数据结构进行实现
        # 示例：
        # return self.params.alpha0 * state['income'] + self.params.alpha1 * state['houseprice'] + ...
        self.params.gamma1 * state['income']
        cost = self.params.gamma1 * state['income'] + self.params.gamma2 * state['houseprice'] + self.params.gamma3 * state['rent'] + self.params.gamma4 * state['mortgage'] + self.params.gamma5 * state['utility']
        return -cost