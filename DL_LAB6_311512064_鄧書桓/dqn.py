'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.optim as optim

class ReplayMemory:
    # 定義了一個名為__slots__的類別屬性，其值為一個包含一個字串'buffer'的列表。
    # 這個屬性的作用是限制該類別實例的屬性，只有在__slots__中列出的屬性名稱才能被定義和使用。
    # 在這個例子中，該類別實例只能有一個名為buffer的屬性。這樣做可以減少實例的記憶體使用，提高程式執行效率。
    __slots__ = ['buffer']

    def __init__(self, capacity):
        # deque 是一種雙向佇列，可以在兩端進行添加和刪除操作
        # maxlen，它指定了 deque 的最大長度。當 deque 的長度超過 maxlen 時，最舊的元素將被自動刪除，
        # 以保持 deque 的長度不超過 maxlen。
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        # 這段程式碼將轉換(transition)的元素轉換為元組(tuple)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        # zip(*transitions)的作用是將多個元組中相同位置的元素打包成一個新的元組。
        # *符號是用來解包列表的，它可以將列表中的元素一一取出
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=32):
        super(Net, self).__init__()
        ## TODO ##
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
        

    def forward(self, x):
        ## TODO ##
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
        


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        # 哲丸用adam且 no amsgrad !!!!!!!
        # 在Adam優化器中，過去所有梯度的平方和都被用來計算調整的學習率，這可能會導致學習率過大或過小。
        # amsgrad通過保留過去所有梯度的平方和的最大值，來避免這個問題。這樣可以確保學習率不會過大或過小，
        # 從而提高模型的收斂速度和穩定性。
        # self._optimizer = optim.AdamW(self._behavior_net.parameters(), lr=args.lr, amsgrad=True)
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr)
        
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq
        self.device = args.device
        self.seed = args.seed

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
         ## TODO ##

        rnd = random.random()
        if rnd < epsilon:
            return np.random.randint(action_space.n)
        else:           
            state = torch.from_numpy(state).float().unsqueeze(0).cuda()
            with torch.no_grad():
                # actions_value = self._behavior_net.forward(state)#state as input out put is action
                action_test = self._behavior_net(state).max(1)[1].view(1, 1).item()
            # action = np.argmax(actions_value.cpu().data.numpy()) #take max q action as action
            action = action_test
            
        return action

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO ##


        q_value = self._behavior_net(state).gather(1, action.long())
        with torch.no_grad():
            q_next= self._target_net(next_state).max(1)[0]
            # view let q_target be same dim with q_value
            q_target = (q_next.view(self.batch_size, 1) * gamma) + reward
        
        # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        # .unsqueeze(1)將向量的維度從一維擴展到二維
        loss = criterion(q_value, q_target)

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        # 將 policy_net 的參數梯度值限制在 -5 到 5 之間。這個函式可以避免梯度爆炸的問題
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0 #不確定這是啥???

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        # 用于初始化随机数生成器的值，它决定了在环境中使用随机数时产生的随机序列。
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        
        while True:
            # env.render()
            action_space = env.action_space
            total_steps = 0 
            action = agent.select_action(state, epsilon, action_space)
            epsilon = max(epsilon * args.eps_decay, args.eps_min)
            next_state, reward, done, _ = env.step(action)
            agent.append(state, action, reward, next_state, done) #為啥需要append???

            state = next_state
            total_reward += reward
            total_steps += 1

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                rewards.append(total_reward)
                print('n_episode: {}\ttotal_reward: {}'.format(n_episode, total_reward))
                break
        
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    # 使用 argparse 模組中的 ArgumentParser 類別建立一個命令列介面的解析器，description 參數指定了解析器的描述，
    # 這裡使用 __doc__ 來作為描述，它會使用該程式檔案的 docstring 來當做描述文字。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=100, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    # 用於將訓練過程中的數據寫入 TensorBoard 中以進行視覺化。
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
