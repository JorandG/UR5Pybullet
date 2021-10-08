import random
import gym
import gym_pybullet 
import numpy as np
from time import sleep
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

EPISODES = 100

x = []
y = []

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9955
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(192, input_dim=self.state_size, activation='relu'))
        model.add(Dense(192, activation='relu'))
        model.add(Dense(192, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("explore")
            return random.randrange(self.action_size)
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(state)
            print("learn action taken :" + str(np.argmax(act_values[0])))
            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, (batch_size))
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
                print("done")
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
 
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #print("explore")

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make("ur5reach-v0")
    state_size = env.observation_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    agent.load("ur5reach-dqn18.h5")
    done = False
    batch_size = 64

    for e in range(EPISODES):
        target1 = env.generate_target()
        state, action, a1_prec, a2_prec, a3_prec, a4_prec = env.reset(target1)
        state = np.reshape(state, [1, state_size])
        for time in range(100):
            print("time :" + str(time))
            action = agent.act(state)
            next_state, reward, done = env.step(action, target1, a1_prec, a2_prec, a3_prec, a4_prec)
            if done:
                reward = reward 
            else:
                reward = -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                x.append(e)
                y.append(time)
                break
            if len(agent.memory) > 8*batch_size:
                agent.replay(batch_size)
        #Uncomment to save the h5 file 
        #if e % 10 == 0:
        #   agent.save("ur5reach-dqn.h5")

plt.plot(x, y) #Plot the evolution of the training
plt.show()
plt.savefig()
