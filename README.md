## Project Details

For this project, you will work with the Reacher environment.

![Unity ML-Agents Reacher Environment](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, there are two separate versions of the Unity environment:

The first version contains a single agent.
The second version contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

## Getting Started

In order to install all the needed dependencies run the following command in your virtual environment:
`pip -q install ./python`

You will also need to uncompress the Unity environment. Extract the content into the root folder of this repo.

## Instructions

In order to start you have to open the file `Continuous_Control.ipynb` in a Jupyter Notebook.

