import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List


def epsilon_greedy(expected_rewards: np.ndarray, num_draws: np.ndarray,
                   **kwargs):
    non_greedy = np.random.uniform() <= kwargs.get('epsilon', 0)
    max_reward_arms = np.where(expected_rewards == expected_rewards.max())[0]
    if non_greedy:
        arm = np.random.choice(len(expected_rewards))
    else:
        arm = np.random.choice(max_reward_arms)
    return arm


def optimistic_initial_values(expected_rewards: np.ndarray,
                              num_draws: np.ndarray, **kwargs):
    optimistic_rewards = expected_rewards + kwargs.get(
        'optimistic_initial_reward', 0)
    return epsilon_greedy(optimistic_rewards, num_draws)


def upper_confidence_bound(expected_rewards: np.ndarray, num_draws: np.ndarray,
                           **kwargs):
    if 0 in num_draws:
        return np.random.choice(np.where(num_draws == 0)[0])
    ucbs = np.zeros_like(expected_rewards)
    for i in range(len(expected_rewards)):
        ucbs[i] = expected_rewards[i] + kwargs.get(
            'ucb_weight', 0.1) * np.sqrt(
                np.log(num_draws.sum()) / num_draws[i])

    max_ucb_arms = np.where(ucbs == ucbs.max())[0]
    return np.random.choice(max_ucb_arms)


POLICIES = {
    'greedy': epsilon_greedy,
    'epsilon_greedy': epsilon_greedy,
    'optimistic_initial_values': optimistic_initial_values,
    'upper_confidence_bound': upper_confidence_bound
}


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
    policy: str = 'greedy'
    learning_rate: float = 0.0
    draws: List[int] = field(default_factory=list)
    rewards: List[int] = field(default_factory=list)
    expected_rewards: np.ndarray = np.zeros(10)
    num_draws: np.ndarray = np.zeros(10)
    epsilon: float = 0.0
    optimistic_initial_reward: float = 0.0
    ucb_weight: float = 0.1

    def __post_init__(self):
        self.expected_rewards = np.zeros(self.testbed.num_arms)
        self.num_draws = np.zeros(self.testbed.num_arms)
        if self.policy not in POLICIES:
            raise ValueError(
                f'''Policy {self.policy} is not available. Choose one of {', '.join(POLICIES.keys())}'''
            )

    def choose_arm(self) -> int:
        return POLICIES[self.policy](
            expected_rewards=self.expected_rewards,
            num_draws=self.num_draws,
            epsilon=self.epsilon,
            optimistic_intial_reward=self.optimistic_initial_reward,
            ucb_weight=self.ucb_weight)

    def draw(self) -> (int, float):
        arm = self.choose_arm()
        reward = self.testbed.draw_arm(arm)
        self.num_draws[arm] += 1
        lr = (1 / self.num_draws[arm]
              ) if self.learning_rate == 0 else self.learning_rate
        self.expected_rewards[arm] += lr * (reward -
                                            self.expected_rewards[arm])
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


def average_multiple_runs(epsilon=0.0, runs=1000):
    run_rewards = []
    for r in range(runs):
        b = Bandit(TestBed(),
                   'optimistic_initial_values',
                   optimistic_initial_reward=5)
        b.run_episode(length=1000)
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
