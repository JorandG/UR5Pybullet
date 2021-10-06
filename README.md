# UR5 with OpenAi Gym in Pybullet
 

  This project aims to control a UR5 using reinforcement learning. It allows to have an autonomous robot that learns from interaction with its environment. The simulation engine used is Pybullet which easy to use and open sourced. 
  
A first version with Q-Learning is better to use if you start in machine learning. Then the Deep-Q-Learning with a neural network can be used for a more accurate simulation. An OpenAi Gym environment has been created that enables to use all precious functionalities that Gym gives for machine learning. Different tasks can be set up. A reach task is detailed, the end effector has to reach a target with the minimum number of steps.

# Table of Contents

- [Simulation](#simulation)
- [Reinforcement Learning](#reinforcement-learning)
- [Pybullet](#pybullet)
- [OpenAi Gym](#openai-gym)
- [Installation](#installation)

# Simulation

## Overview of the task

The environment ur5reach_env.py is designed for a simple task of reaching a target in front of the UR5 using reinforcement learning. For that the robot only knows the position of the end effector and the distance between it and the target (we assume we can "easily" get this information with a camera). The robot should then train in order to know the environment and finally reach the target with the minimum number of steps. 

## State and action space

To speed up the training a discrete State Space has been set up. Only the interesting joint angles were selected for the study and experiments.

<img width="677" alt="UR_presentation1 (1)" src="https://user-images.githubusercontent.com/91953623/136206583-0ffa8d5d-9d77-44e8-9597-b625042fc994.png">

The possible angles for the 4 joints are as followed:

• Shoulder_pan_joint = [1.8 1.56 1.32 1.08 0.84 0.6 0.36 0.12 -0.12 -0.36 -0.6 -0.84 -1.08 -1.32 -1.56 -1.8]

• Shoulder_lift_joint = [-0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9 -1.1 -1.2 -1.3]

• Elbow_joint = [0.8 0.935 1.07 1.205 1.34 1.475 1.61 1.745 1.88 2.015 2.15]

• Wrist_1_joint = [-0.5 -1.0 -1.5 -2.0]

The state space is then 16 x 10 x 11 x 4 = 7040 states. The action space is 16 + 10 + 11 + 4 = 41 actions.


# Reinforcement Learning

The main concepts of RL are: the environment, the state, actions, reward and penalties and the policy. Let’s explain it with a concrete example. We basically want to teach a robotic arm to reach an object on a table without knowing anything else than the positions of its end-effector and the joint angles.

• The agent is the robotic arm itself.

• Then the environment which could be reduced to the room in which the robot is, the table, and the objects. This is where the robot will evolve.

• The first state would be the robot at the starting position then the other state will be the end effector that ”grasp” the object with magnetization and the final state would be the object placed at the desired position on the table.

• The transition between each state is made through actions. Here actions are made by moving the different joints of the 6 DOF arm.

• The reward could be for example +1 if the robot goes closer to the assembly element penalties are given when it goes in the wrong directions and too far from the target.

• Finally the policy is the strategy of choosing an action given a state in expectation of better outcomes.

## Q-Learning

You should first learn about Q-Learning. Essentially, Q-Learning lets the agent use the environment’s rewards to learn, over time, the best action to take in a given state.

Q_values are initialized to an arbitrary value, and as the agent exposes itself to the environment and receives different rewards by executing different actions, the Q_values are updated using the Bellman equation:

Q(s, a) = r(s, a) + γ*max Q(s′, a) 

The above equation states that the Q_value yielded from being at state s and performing action a is the immediate reward r(s, a) plus the highest Q_value possible from the next state s’, γ here is the discount factor that controls the contribution of rewards further in the future.

### Q-table

In Q Learning a Q table is used as the memory. The table is composed of the actions and states and is updated with the Bellman equation at every step. The table is initialized with zeros.

<img width="822" alt="Q_table" src="https://user-images.githubusercontent.com/91953623/136204131-6c27e4dd-9323-4780-9582-9c1eacaf5252.png">

After multiple episodes of training the Q table is filled with coefficient through the Bellman equation.

<img width="822" alt="Q_table_filled" src="https://user-images.githubusercontent.com/91953623/136204268-4c6f6bc6-d71d-4712-a21d-9cb84b7c165c.png">

When the robot exploits the Q table it looks at the line for a certain state. For example in the state 3, Shoulder_pan_joint = 1.32, Shoulder_lift_joint = -0.3, Elbow_joint = 0.8 and Wrist_1_joint = -0.5 the optimal action according to the Q table is to move the Wrist_1_Joint to -2.0.

<img width="820" alt="Q_table_choix" src="https://user-images.githubusercontent.com/91953623/136204399-ff9065ee-e055-4242-a8bc-fc8bcabdf9a7.png">

### Training the agent 

First, we’ll initialize the Q-table to a matrix of zeros. Then the training algorithm is created, it will update this Q-table as the agent explores the environment over one thousand episodes.

In the first part of while not done, it is decided whether to pick a random action or to exploit the already computed Q-values. This is done simply by using the epsilon value and comparing it to the random.uniform(0, 1) function, which returns an arbitrary number between 0 and 1.

The chosen action is executed in the environment to obtain the next_state and the reward from performing the action. After that, the maximum Q_value is calculated for the actions corresponding to the next_state, and with that, the Q_value can easily be updated to the new_q_value. 

One epoch is completed once the end-effector reaches the target with a minimum distance.

For more informations about Q-Learning and OpenAi Gym I recommand this article:

[Q-Learning using OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym)
 
## Deep-Q-Learning

In Deep-Q-learning, a neural network is used to approximate the Q_value function using [Keras](https://keras.io) which is a library of [TensorFlow](https://github.com/tensorflow/tensorflow). The state is given as the input and the Q_value of all possible actions is generated as the output. In the DQN.py we used a Sequential model with a ReLu activation fonction which is the most common one. In the case of the study the first layer is of the size of the state_size which is the number of possible moves for the robot, so 4.

For more informations about Deep-Q-Learning and OpenAi Gym you can read this article:

[Deep Q-Learning using OpenAI Gym](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python)

### Network 

![neuron-nn](https://user-images.githubusercontent.com/91953623/136211975-4c9de1ea-83aa-434c-845e-2a4c8ae944cd.PNG)


The network has three layers, an input layer with three input units, a hidden layer, and an output layer consisting of two output units. The number of circles in each layer indicates the dimensions of the corresponding layers. The circles represent neurons of the network, and arrows represent the connections and data between the neurons of the network.

#### Weight and bias 

Weights and biases (commonly referred to as w and b) are the learnable parameters of a machine learning model. Neurons are the basic units of a neural network. In a DNN, each neuron in a layer is connected to each neuron in the next layer. When the inputs are transmitted between neurons, the weights are applied to the inputs along with the bias.

Weights control the signal (or the strength of the connection) between two neurons. In other words, a weight decides how much influence the input will have on the output.

Biases, which are constant, are an additional input into the next layer that will always have the value of 1. 

These are the parameters stored in the .h5 file that you get after the training of your agent. 

### Experience Replay

Experience Replay stores experiences including state transitions, rewards, and actions, which are necessary data to perform Q learning, and makes mini­batches to update neural networks. This technique expects the following merits:

• reduces the correlation between experiences in updating DQN 

• increases learning speed with mini-batches

• reuses past transitions to avoid catastrophic forgetting

### Target Network

Since the same network is calculating the predicted value and the target value, there could be a lot of divergence between these two.

A separate network to estimate the target could also be used. This target network has the same architecture as the function approximator but with frozen parameters. For every C iterations (a hyperparameter), the parameters from the prediction network are copied to the target network. This leads to more stable training because it keeps the target function fixed.

# Pybullet

  [Pybullet Quickstart guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)
  
# OpenAi Gym

  [OpenAI Gym repo](https://github.com/openai/gym)
  
# Installation




