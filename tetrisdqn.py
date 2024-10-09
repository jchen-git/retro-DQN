import os
import random
import gym
import retro
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from agent import Agent
from preprocessing import preprocess, stack_frame

# Set path for custom ROMs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_STATES = [os.path.join(root, name)
             for root, dirs, files in os.walk(os.path.join(SCRIPT_DIR, 'custom_integrations'))
             for name in files
             if name.endswith(".state")]

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, 'custom_integrations')
)

env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)
#env = gym.make("CartPole-v1")
num_actions = env.action_space.n

# CartPole Resolution is (600 x 400)
IMAGE_CROP = (35, 204, 85, 170)
#IMAGE_CROP = (150, 350, 50, 550)
#IMAGE_CROP = (1, -1, 1, -1)
INPUT_SHAPE = (1, 96, 96)
#INPUT_SHAPE = 4

agent = Agent(INPUT_SHAPE, num_actions, "tetris", training=True)
#agent = Agent(INPUT_SHAPE, num_actions, "cart", training=True)
is_training = True
rewards_per_episode = []
epsilon_history = []
best_reward = -1.0
best_reward_episodes = []

#agent.policy_net.load_state_dict(torch.load(agent.MODEL_FILE, weights_only=True))

log_message=f"{datetime.now()}: Training..."
print(log_message)
with open(agent.LOG_FILE, 'w') as file:
    file.write(log_message + '\n')

for episode in range(agent.epoch):
    env.load_state(random.choice(SAVE_STATES))
    state = env.reset()

    done = False
    episode_reward = 0.0

    frame = preprocess(state, IMAGE_CROP, agent.image_resize)
    #frame = preprocess(env.render("rgb_array"), IMAGE_CROP, agent.image_resize)
    frames = stack_frame(None, frame, True)
    state = torch.tensor(frames, dtype=torch.float32, device="cuda").unsqueeze(0)

    # See frame after preprocessing
    # plt.figure()
    # plt.imshow(frames[0], cmap="gray")
    # plt.title('Original Frame')
    # plt.show()

    while not done:
        env.render()
        action = agent.act(state)
        obs, rew, done, info = env.step(agent.actions[action.item()])
        #obs, rew, done, info = env.step(action.item())
        episode_reward += rew
        frame = preprocess(obs, IMAGE_CROP, agent.image_resize)
        #frame = preprocess(env.render("rgb_array"), IMAGE_CROP, agent.image_resize)
        frames = stack_frame(frames, frame, False)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(frames, dtype=torch.float32, device='cuda').unsqueeze(0)

        rew = torch.tensor([rew], device='cuda')

        agent.replay_memory.append(state, action, rew, next_state)

        if (len(agent.replay_memory)) > agent.batch_size:
            agent.optimize()

        state = next_state

    #print("Reward at end of episode:" + str(episode_reward))

    if is_training:
        if episode_reward > best_reward:
            log_message= f"{datetime.now()}: New best reward: {episode_reward:0.1f} at episode {episode}"
            print(log_message)
            with open(agent.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

            torch.save(agent.policy_net.state_dict(), agent.MODEL_FILE)
            best_reward = episode_reward
            # Track rewards per episode
    rewards_per_episode.append(episode_reward)

    epsilon_history.append(agent.epsilon)

# Save plots
fig = plt.figure(1)
durations_t = torch.tensor(rewards_per_episode, dtype=torch.float)
# Plot average rewards (Y-axis) vs episodes (X-axis)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.plot(durations_t.numpy())
if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

# Save plots
fig.savefig(agent.GRAPH_FILE)
plt.close(fig)

env.close()

log_message=f"{datetime.now()}: Training Complete"
print(log_message)
with open(agent.LOG_FILE, 'a') as file:
    file.write(log_message + '\n')