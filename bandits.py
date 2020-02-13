#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
from box import Box
from typing import List


#%%
@dataclass
class TestBed:
    reward_mean: float = 0.
    reward_std: float = 1.
    num_arms: int = 10
    seed: int = 0
    mean_rewards: np.ndarray = np.random.normal(0, 1, 10)

    def __post_init__(self):
        self.seed = np.random.choice(1000000)
        np.random.seed(self.seed)
        self.mean_rewards = np.random.normal(self.reward_mean, self.reward_std,
                                             self.num_arms)

    def draw_arm(self, arm_idx: int):
        mu = self.mean_rewards[arm_idx]
        return np.random.normal(mu, self.reward_std)


@dataclass
class Bandit:
    testbed: TestBed
    epsilon: float = 0.0
    draws: List[int] = field(default_factory=list)
    rewards: List[int] = field(default_factory=list)
    expected_rewards: np.ndarray = np.zeros(10)
    num_draws: np.ndarray = np.zeros(10)

    def __post_init__(self):
        self.expected_rewards = np.zeros(self.testbed.num_arms)
        self.num_draws = np.zeros(self.testbed.num_arms)

    def choose_arm(self) -> int:
        non_greedy = np.random.uniform() <= self.epsilon
        max_reward_arms = np.where(
            self.expected_rewards == self.expected_rewards.max())[0]
        if non_greedy:
            arm = np.random.choice(self.testbed.num_arms)
        else:
            arm = np.random.choice(max_reward_arms)
        return arm

    def draw(self) -> (int, float):
        arm = self.choose_arm()
        reward = self.testbed.draw_arm(arm)
        self.num_draws[arm] += 1
        self.expected_rewards[arm] += (
            reward - self.expected_rewards[arm]) / self.num_draws[arm]
        self.rewards.append(reward)
        self.draws.append(arm)
        return (arm, reward)

    def run_episode(self, length: int = 2000):
        for i in range(length):
            arm, reward = self.draw()

    def reset(self):
        self.draws = []
        self.rewards = []
        self.expected_rewards = np.zeros(self.testbed.num_arms)
        self.num_draws = np.zeros(self.testbed.num_arms)


#%%
def average_multiple_runs(epsilon=0.0, runs=1000):
    run_rewards = []
    for r in range(runs):
        b = Bandit(TestBed(), epsilon)
        b.run_episode()
        run_rewards.append(b.rewards)
    return np.vstack(run_rewards).mean(axis=0)


def compare_greedy_levels(*epsilons: float):
    plt.figure(figsize=(12, 10))
    for e in epsilons:
        mean_rewards = average_multiple_runs(e)
        plt.plot(range(len(mean_rewards)), mean_rewards, label=e)
    plt.ylim(-0.5, 2.5)
    plt.legend()
    plt.show()
