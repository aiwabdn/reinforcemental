import numpy as np
import matplotlib.pyplot as plt

sz_bandit = 10
sz_testbed = 2000
timesteps = 1000
eps = [0.01,0.1,0] # probability of choosing random arm

# mean rewards for one bandit
testbed = np.random.normal(size=(sz_testbed, sz_bandit))

# create array to maintain average rewards incrementally over time
r_mean = np.zeros((sz_testbed,sz_bandit))
cum_reward = np.zeros((3,timesteps))

# at each timestep select the arm with max rewards and update r_mean
for e in range(len(eps)):
    # reset counts of selection and expected rewards
    r_select_counts = np.zeros((sz_testbed,sz_bandit))
    r_mean = np.zeros((sz_testbed, sz_bandit))
    for i in range(timesteps):
        for ix,row in enumerate(r_mean):
            #select arm
            arm = np.argmax(row) if np.random.uniform() <= (1 - eps[e]) else np.random.choice(np.arange(sz_bandit))
            # update arm selection count
            r_select_counts[ix,arm] += 1
            #get reward for that arm
            R_t = np.random.normal(testbed[ix,arm], 1)
            cum_reward[e,i] += R_t
            #update mean reward for that arm
            row[arm] += (R_t - row[arm])/r_select_counts[ix,arm]

print(r_select_counts)
print(cum_reward)

# plot average rewards over time
for e in range(len(eps)):
    plt.plot(cum_reward[e]/sz_testbed, label = 'e='+str(eps[e]))
#plt.legend()
#plt.show()

# the learning method with the lowest e-greedy learns slowly but constantly grows over time.
# next we try to learn with a constant step size instead of time varying step size. The above method works well when the reward distributions are stationary as the expected values of rewards can be calculated by the average of the past rewards. In case of non-stationary distributions the most popular technique is to have a constant step size and use that to take a weighted average of all past rewards of an arm, weighted in a descending order of their recency.
alpha = 0.1 # step size 
for e in range(len(eps)):
    # reset counts of selection and expected rewards
    r_select_counts = np.zeros((sz_testbed,sz_bandit))
    r_mean = np.zeros((sz_testbed, sz_bandit))
    for i in range(timesteps):
        for ix,row in enumerate(r_mean):
            #select arm
            arm = np.argmax(row) if np.random.uniform() <= (1 - eps[e]) else np.random.choice(np.arange(sz_bandit))
            # update arm selection count
            r_select_counts[ix,arm] += 1
            #get reward for that arm
            R_t = np.random.normal(testbed[ix,arm], 1)
            cum_reward[e,i] += R_t
            #update mean reward for that arm
            row[arm] = alpha*(1-alpha)*R_t + (1-alpha)*row[arm]

# UCB method, selection of arm with preference to ones with max variance
r_select_counts = np.zeros((sz_testbed,sz_bandit))
r_mean = np.zeros((sz_testbed, sz_bandit))
c = 0.5 #parameter to control variance in UCB
cum_reward = np.zeros(timesteps)
for i in range(timesteps):
    for ix,row in enumerate(r_mean):
        #select arm
        arm = np.argmax(row + c * np.sqrt(np.log(i+1)/r_select_counts[ix]))
        # update arm selection count
        r_select_counts[ix,arm] += 1
        #get reward for that arm
        R_t = np.random.normal(testbed[ix,arm], 1)
        cum_reward[i] += R_t
        #update mean reward for that arm
        row[arm] += (R_t - row[arm])/r_select_counts[ix,arm]

plt.plot(cum_reward/sz_testbed, label='UCB')
#plt.legend()
#plt.show()

# Gradient Bandit algorithm. 
# Maintain a set of preferences for each bandit. At each timestep, select arm with highest preference, get reward, and update preferences for all arms. The choice of baseline to compare the reward received at each step to does not affect the selection of arms. That is dictated only by the relative probabilities of selection of the arms which is calculated by softmax over the preferences for a bandit. The choice of base line does affect the performance of the learning process i.e. the convergence rate. The choice of the average reward till the present time step is a simple and effective baseline. It could be any scalar independent of the choice of the arm. The preference update is akin to gradient ascent for the selected arm. The other arms get updates in the opposite direction. In case the received reward is higher than the baseline the preference for that arm goes up while those of the other arms go down.
def softmax(row):
    e = np.exp(row)
    return(e/sum(e))
for alpha in [0.1,0.4]:
    baseline = np.zeros(sz_testbed)
    H = np.zeros((sz_testbed, sz_bandit)) # set of preferences for each arm in all bandits
    cum_reward = np.zeros(timesteps)
    for t in range(timesteps):
        for ix,row in enumerate(H):
            # calculate softmax probabilities for each arm
            pi_t = softmax(row)
            # select arm and get reward
            arm = np.argmax(pi_t)
            R_t = np.random.normal(testbed[ix], 1)[arm]
            # update baseline
            baseline[ix] += (R_t - baseline[ix])/(t+1)
            # update preferences
            H[ix] -= alpha * (R_t - baseline[ix]) * pi_t
            H[ix,arm] += alpha * (R_t - baseline[ix])
            cum_reward[t] += R_t
    plt.plot(cum_reward/sz_testbed, label='GB'+str(alpha))

plt.legend()
plt.show()

