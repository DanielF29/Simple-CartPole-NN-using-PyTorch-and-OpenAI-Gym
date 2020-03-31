# Simple CartPole NN (DQN) trained by Reinforcement Learning using PyTorch and OpenAI-Gym
Simple CartPole controlled by a NN using PyTorch and OpenAI Gym

This code is base on a PyTorch tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html and on the adaptation of that code found at https://discuss.pytorch.org/t/help-for-dqn-example/13263

An image of a result from the code on this repository:
![](https://github.com/DanielF29/Simple-CartPole-NN-using-PyTorch-and-OpenAI-Gym/blob/master/20200329_1733_SimpleCartPole_trainningResult.png)

The vertical axis is the "Duration" of each simulation done on the CartPole environment ('CartPole-v0'), you can get started with GYM at https://gym.openai.com/ , and the horizontal axis is the number of simulations the Neural Net (NN) did.

As you can see the results from the training are not that consistent, but definitely the NN is learning and doing quite well some times.

Note: The training does not render the environment until it's finished and only renders the last state of the last run of the simulations, you could render the environment while training but it will take longer for the same training, so I will recommend adding some condition to render only the last run or the last 3 runs of the simulations used for training if you want to see the behavior acquired... also the code at the moment does not save the trained NN for future runs (I will add that part at some point in the future...but do not count on that been soon)
