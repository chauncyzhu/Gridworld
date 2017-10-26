# coding=utf8
"""
网格世界的代码实现
"""
import random
import numpy as np
import collections
class GridWorld():
    """
    创建gridworld的环境
    """
    def __init__(self, map_file_name):
        """
        初始化函数
        """

        # 定义action集合，是预先设定好的，由于网格只有上下左右四个动作，因此为[1,0],[0,1],[-1,0],[0,-1]
        self._action = [[1,0], [0,1], [-1,0], [0,-1]]

        # 定义网格世界，需要读取网格世界的文件
        self._map = self._readmap(map_file_name)
        self._size = self._map.shape
        self._st = None  # 初始化时的当前状态
    
    def get_map_size(self):
        return self._size

    def get_curr_state(self):
        return self._st


    def get_action_size(self):
        return len(self._action)

    def _readmap(self, file_name):
        grid = []
        with open(file_name) as f:
            for line in f:
                grid.append(list(line.strip().upper()))
        
        return np.asarray(grid)


    def regist_state(self, state=None):
        """
        注册状态，如果需要注册的状态为None，则设为[0,0]，否则检查一下该状态是否符合规范，符合则更新当前state
        """
        st = [0, 0] if state == None else state
        
        """
        检查状态是否符合规范
        """
        if self.check_state(st) == 'O':   # 如果仍然还是位于非陷阱、非目标位置
            self._st = np.asarray(st)
        else:  # 如果不是则报错，但是如果进入了陷阱和目标该怎么办？
            return ValueError('wrong state input')
            
    

    def check_state(self, state):
        """
        检查状态是否符合规范，如果当前state超出了范围就不行，如果可以就取出当前状态对应的标记
        N表示未知的state，表明超出了范围，F表示错误的state
        """
        if isinstance(state, collections.Iterable) and len(state) == 2:
            if state[0] < 0 or state[1] < 0 or state[0] >= self._size[0] or state[1] >= self._size[1]:
                return 'N'
            # 取出状态对应的标记
            return self._map[state]

        else:  # 错误状态
            return 'F'


    def next_state(self, ac):
        """
        接受action，更新当前state，这里的action是采取序号
        FIXME: 为什么当为O或者G时才会更新状态？
        """
        new_st = self._st + self._action[ac]

        # 检查当前状态，下面的判断语句说明了只位于一个状态，不能同时拥有多个状态
        ch_new_st = self.check_state(new_st)
        if ch_new_st == 'N' or ch_new_st == 'H':  # 如果越界或者掉进陷阱则扣五分
            self._st = new_st
            return -5
        elif ch_new_st == 'O':   # 如果仍然是普通状态则扣1分
            self._st = new_st        
            return -1
        elif ch_new_st == 'G':  # 如果到达了目标
            self._st = new_st                    
            return 50
        else:
            return ValueError('wrong action input')


def epsilon_greedy(epsilon, ac_size, state, Q):
    """
    产生贪心策略
    FIXME: 为什么要进行贪心策略？
    """
    if random.random() < epsilon:
        action = random.randint(ac_size)  # 产生一个随机策略
    else:  # 否则根据当前state由Q值产生最大策略
        action = np.where(Q[state[0], state[1], :] == Q[state[0], state[1], :].max())   # 产生最大的Q值对应的action

    return action

def random_action(ac_size):
    return random.randint(ac_size) - 1

def max_q(state, Q):
    """
    根据state取出Q中对应最大的action所在的Q值
    """
    return Q[state[0], state[1], :].max()
    

def Qlearning():
    """
    开始进行Q learning，实际上每次更新的时候，是对Q table中的state和action对应的值进行更新
    Q learning最终的目的是为了使每个state下采取Q值最高的action
    FIXME:如何选择一个action？
    FIXME:是否需要设计experiments + episode的方式，episode用来每次训练，同时设置终止条件如掉进陷阱或者达到目标，experiments用来进行多次episode
    """
    # 设置一些参数
    learning_rate = 0.1  # 学习率
    discount_rate = 0.1   # 折扣率
    experiments = 10000   # 片段，也是时间的长度
    epsilon = 0.1

    # 加载一个网格世界
    grid_world = GridWorld('file_name')
    map_size = grid_world.get_map_size()  # 网格世界的大小
    action_size = grid_world.get_action_size()

    # 注册一个初始化状态
    init_state = grid_world.regist_state()

    # 构建一个Q table来存储Q值
    q_table = np.zeros(shape=(map_size[0],map_size[1],action_size))

    # offline train
    for i in range(experiments):   # n th episode
        print("now ", i, " experiments")
        # 下面实验随机产生action，然后来更新Q值
        old_state = grid_world.get_curr_state()  # 为进行action前的状态
        action = random_action(action_size)  # 选取一个动作
        q_table[old_state[0], old_state[1], action] *= 1-learning_rate
        reward = grid_world.next_state(action)   # 获取采取action后的reward，同时通过next_state将对应的state赋给了curr_state
        curr_state = grid_world.get_curr_state()
        # 更新q值
        q_table[old_state[0], old_state[1], action] += learning_rate*(reward + discount_rate*max_q(curr_state, q_table)) 
        
    
    # 结束
    print("Q table value:", q_table)


if __name__ == '__main__':
    Qlearning()