"""Training the agent"""
import gym 
import gym_pybullet 
from time import sleep
import numpy as np
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
import h5py
import pybullet as p

env = gym.make("ur5reach-v0")

q_table = np.zeros([7040, 42]) #Initialize the Q Table with zeros the Q Table is (state size, action size)

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.4

# For plotting metrics
all_epochs = []
all_penalties = []

x = []
y = []
#action = 12

for i in range(1, 200):
    
    #print(q_table)
    alpha = 0.5-(i*0.0015)
    #print(alpha)
    gamma = 1-(i*0.001)
    #print(gamma)
    epsilon = 0.6-(i*0.0015)
    #print(epsilon)
    target1 = env.generate_target()
    state, action, a1_prec, a2_prec, a3_prec, a4_prec = env.reset(target1)

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        
        state = env.stateQLearning(action, a1_prec, a2_prec, a3_prec, a4_prec)
        if random.uniform(0, 1) < epsilon:
            action, a1_prec, a2_prec, a3_prec, a4_prec = env.action_sample(a1_prec, a2_prec, a3_prec, a4_prec) # Explore action space
            print("explore")
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
            print("learn")
                    
            
        next_state, reward, done = env.stepQLearning(action, target1, a1_prec, a2_prec, a3_prec, a4_prec)
        env.step_simu()
        print(state)
        print(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -5:
            penalties += 1
        
        epochs += 1
        if epochs > 1000:
            done=True

        print("epochs ------:" + str(epochs))

    print("Episode : " + str(i) )    
    print("Num epochs : " + str(epochs))
    print("Num penalties :" + str(penalties))
    x.append(i)
    y.append(epochs)


print("Training finished.\n")
plt.plot(x, y)
plt.show()

np.savetxt('/home/user/file/your_output_file.txt', q_table, delimiter="\t") 

np.save('q_table.npy', q_table)
