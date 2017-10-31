# coding=utf8
"""
网格世界的代码实现
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import collections

INIT_STAT = [2,0]

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

    def get_goal_coordinate(self):
        return change_coordinate(np.where(self._map == 'G'), self._size)


    def get_action_size(self):
        return len(self._action)

    def _readmap(self, file_name):
        grid = []
        with open(file_name) as f:
            for line in f:
                grid.append(list(line.strip().upper()))
        return np.asarray(grid)

    def get_reward(self, ch_st):
        """
        以随机概率获取reward
        :return:
        """
        if ch_st == 'O':
            # return random.choice([-12, 10])
            return -1
        if ch_st == 'G':
            return 5


    def regist_state(self, state=None):
        """
        注册状态，如果需要注册的状态为None，则设为[0,0]，否则检查一下该状态是否符合规范，符合则更新当前state
        """
        st = INIT_STAT if state == None else state
        """
        检查状态是否符合规范
        """
        if self.check_state(st) == 'O':   # 如果仍然还是位于非陷阱、非目标位置
            self._st = np.asarray(st)
            return self.get_reward(self.check_state(st))
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
            return self._map[tuple(state)]

        else:  # 错误状态
            return 'F'


    def next_state(self, ac):
        """
        接受action，更新当前state，这里的action是采取序号
        FIXME: 为什么当为O或者G时才会更新状态？而不是位于正常以及目标的时候不更新当前状态？
        """
        new_st = self._st + self._action[ac]

        # 检查当前状态，下面的判断语句说明了只位于一个状态，不能同时拥有多个状态
        ch_new_st = self.check_state(new_st)
        if ch_new_st == 'N' or ch_new_st == 'H':  # 如果越界或者掉进陷阱则扣五分，且不更新状态
            # self._st = new_st
            return -5
        elif ch_new_st == 'O':   # 如果仍然是普通状态则扣1分
            self._st = new_st
            return self.get_reward(ch_new_st)
        elif ch_new_st == 'G':  # 如果到达了目标
            self._st = new_st
            return self.get_reward(ch_new_st)
        else:
            return ValueError('wrong action input')


def epsilon_greedy(epsilon, ac_size, state, Q):
    """
    产生贪心策略
    FIXME: 为什么要进行贪心策略？
    """
    if random.random() < epsilon:
        action = random.randint(0, ac_size-1)  # 产生一个随机策略
    else:  # 否则根据当前state由Q值产生最大策略
        action_max = np.where(Q[state[0], state[1], :] == Q[state[0], state[1], :].max())[0]   # 产生最大的Q值对应的action，因为是tuple，所以需要取第一个
        action = int(random.choice(action_max))   # 可能有多个，随机选取一个
    return action


def random_action(ac_size):
    return random.randint(0, ac_size-1)


def max_q(state, Q):
    """
    根据state取出Q中对应最大的action所在的Q值
    """
    return Q[state[0], state[1], :].max()

def change_coordinate(state, map_size):
    """
    更改坐标，因为原先读取数据文件的时候是从map的左上角开始读起
    """
    return [state[1], map_size[0]-1-state[0]]


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
    experiments = 40000   # 片段，也是时间的长度
    episode = 4
    epsilon = 0.5
    max_test_step = 4
    file_name = '2.txt'

    # 加载一个网格世界
    grid_world = GridWorld(file_name)
    map_size = grid_world.get_map_size()  # 网格世界的大小
    action_size = grid_world.get_action_size()

    # 构建一个Q table来存储Q值
    q_table = np.zeros(shape=(map_size[0],map_size[1],action_size))

    # 存储一系列的状态便于后面绘图
    max_q_a = []   # 每个timestep的初始状态最大的q值
    reward_ex = []   # 计算每个time step的平均reward

    # offline train
    for i in range(experiments):   # n th episode
        print("now ", i, " exper2")

        # 注册一个初始化状态
        reward_list = [grid_world.regist_state()]
        init_state = grid_world.get_curr_state()
        max_q_a.append(max_q(init_state, q_table))

        for j in range(episode):
            # 下面实验随机产生action，然后来更新Q值
            old_state = grid_world.get_curr_state()  # 为进行action前的状态
            # action = random_action(action_size)  # 选取一个动作
            action = epsilon_greedy(epsilon, action_size, old_state, q_table)  # 选取一个动作

            q_table[old_state[0], old_state[1], action] *= 1-learning_rate
            reward = grid_world.next_state(action)   # 获取采取action后的reward，同时通过next_state将对应的state赋给了curr_state
            curr_state = grid_world.get_curr_state()
            # 更新q值
            q_table[old_state[0], old_state[1], action] += learning_rate*(reward + discount_rate*max_q(curr_state, q_table))

            # 计算平均reward
            reward_list.append(reward)

        # 计算平均reward
        reward_ex.append(np.array(reward_list).mean())


    # 使用Q table进行测试，加载网格世界
    grid_world_test = GridWorld(file_name)
    map_size = grid_world_test.get_map_size()  # 网格世界的大小
    all_test_state_cor = []
    grid_world_test.regist_state()
    test_state = grid_world_test.get_curr_state()

    # 每一步的action都采取Q table中最大的action
    count = 0
    while True:
        old_state =  grid_world_test.get_curr_state()
        all_test_state_cor.append(change_coordinate(old_state, map_size))
        action = q_table[old_state[0], old_state[1], :].argmax()
        reward = grid_world_test.next_state(action)
        print('reward:',reward)
        print("curr state:",grid_world_test.get_curr_state())

        if count > max_test_step:
            break
        count+=1

    # 绘图
    plt.clf()

    # 绘制max q
    plt.plot(range(experiments), max_q_a)
    plt.show()

    # 绘制平均reward
    plt.plot(range(experiments), reward_ex)
    plt.show()

    # 绘制agent坐标变换的图，这个是test的
    all_test_state_cor = np.array(all_test_state_cor)  # 可以用vstack避免这一步的转换
    plt.plot(all_test_state_cor[:, 0], all_test_state_cor[:, 1])
    goal_pos = grid_world_test.get_goal_coordinate()
    print('test state:', test_state)
    test_pos = change_coordinate(test_state, grid_world_test.get_map_size())
    print('test pos:', test_pos)
    plt.text(goal_pos[0], goal_pos[1], 'G')
    plt.text(test_pos[0], test_pos[1], 'BEGIN')
    plt.xlim([0, grid_world_test.get_map_size()[1]])
    plt.ylim([0, grid_world_test.get_map_size()[0]])
    plt.show()



if __name__ == '__main__':
    Qlearning()