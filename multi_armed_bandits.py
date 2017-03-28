import numpy as np
import matplotlib.pyplot as plt

sz_bandit = 10
sz_testbed = 2000
timesteps = 1000
eps = [0.01,0.1,0] # probability of choosing random arm

# mean rewards for one bandit
r = np.random.normal(0,1,sz_bandit)
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

# UCB method, selection of arm with preference to ones with max variance
r_select_counts = np.zeros((sz_testbed,sz_bandit))
r_mean = np.zeros((sz_testbed, sz_bandit))
c = 0.5 #parameter to control variance in UCB
cum_reward = np.zeros(timesteps)
for i in range(timesteps):
    for ix,row in enumerate(r_mean):
        #select arm
        arm = np.argmax(row + c * np.sqrt(np.log(i)/r_select_counts[ix]))
        # update arm selection count
        r_select_counts[ix,arm] += 1
        #get reward for that arm
        R_t = np.random.normal(testbed[ix,arm], 1)
        cum_reward[i] += R_t
        #update mean reward for that arm
        row[arm] += (R_t - row[arm])/r_select_counts[ix,arm]

plt.plot(cum_reward/sz_testbed, label='UCB')
plt.legend()
plt.show()


