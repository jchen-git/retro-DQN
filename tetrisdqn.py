import os
import random
#import gym
import retro
import torch
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from agent import Agent
from preprocessing import preprocess, stack_frame

# TODO
# Change rewards given such that bottom four rows should be filled
# Create GUI

def create_graphs():
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

    # Save plots
    fig = plt.figure(1)
    durations_t = torch.tensor(score_per_episode, dtype=torch.float)
    # Plot average rewards (Y-axis) vs episodes (X-axis)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Save plots
    fig.savefig(agent.GRAPH_SCORE_FILE)
    plt.close(fig)

# Set path for custom ROMs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SAVE_STATES = [os.path.join(root, name)
             for root, dirs, files in os.walk(os.path.join(SCRIPT_DIR, 'custom_integrations'))
             for name in files
             if name.endswith(".state")]

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, 'custom_integrations')
)

matplotlib.use('Agg')

# Tetris game
env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)
num_actions = env.action_space.n
IMAGE_CROP = (35, 204, 85, 170)
INPUT_SHAPE = (4, 128, 128)
agent = Agent(INPUT_SHAPE,"tetris", training=True)

is_training = True
rewards_per_episode = []
score_per_episode = []
best_reward = -999.0

log_message=f"{datetime.now()}: Training..."
print(log_message)
with open(agent.LOG_FILE, 'w') as file:
    file.write(log_message + '\n')

if os.path.isfile(agent.DATA_FILE):
    with open(agent.DATA_FILE, 'r') as file:
        best_reward = float(file.read())

agent.policy_net.load_state_dict(torch.load(agent.MODEL_FILE, weights_only=True))

for episode in range(agent.epoch):
    env.load_state(random.choice(SAVE_STATES))
    state = env.reset()

    done = False
    episode_reward = 0.0

    frame = preprocess(state, IMAGE_CROP, agent.image_resize)
    frames = stack_frame(None, frame, True)
    state = torch.tensor(frames, dtype=torch.float32, device="cuda").unsqueeze(0)

    while not done:
        total_reward = 0
        current_step = 0
        env.render()

        action = agent.act(state)
        while current_step < (agent.FRAME_SKIPS + 1) and not done:
            obs, rew, done, info = env.step(agent.actions[action.item()])
            total_reward += rew
            current_step += 1
        rew = total_reward
        episode_reward += rew

        frame = preprocess(obs, IMAGE_CROP, agent.image_resize)
        frames = stack_frame(frames, frame, False)

        next_state = torch.tensor(frames, dtype=torch.float32, device='cuda').unsqueeze(0)
        rew = torch.tensor([rew], device='cuda')
        done = torch.tensor(done, device='cuda')

        agent.replay_memory.append(state, action, rew, next_state, done)

        if (len(agent.replay_memory)) > agent.batch_size:
            agent.optimize()

        state = next_state

    # Track rewards per episode
    rewards_per_episode.append(episode_reward)
    score_per_episode.append(int(info['score']))

    if is_training:
        if episode_reward > best_reward:
            log_message= f"{datetime.now()}: New best reward: {episode_reward:0.2f} at episode {episode}"
            print(log_message)
            with open(agent.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

            torch.save(agent.policy_net.state_dict(), agent.MODEL_FILE)
            best_reward = episode_reward

            with open(agent.DATA_FILE, 'w') as file:
                file.write(str(best_reward))

        if episode % 50 == 0:
            log_message = f"{datetime.now()}: Episode {episode} complete"
            print(log_message)
            create_graphs()

create_graphs()

env.close()

log_message=f"{datetime.now()}: Training Complete"
print(log_message)
with open(agent.LOG_FILE, 'a') as file:
    file.write(log_message + '\n')