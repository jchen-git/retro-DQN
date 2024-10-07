import os
import retro
import torch

from agent import Agent
from preprocessing import preprocess

# Set path for custom ROMs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, 'custom_integrations')
)

env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)
num_actions = env.action_space.n

# TODO
# Rename this variable
IMAGE_CROP = (35, 204, 85, 170)
INPUT_SHAPE = (1, 64, 64)

agent = Agent(INPUT_SHAPE, num_actions, "tetris", training=True)
is_training = True
rewards_per_episode = []
epsilon_history = []
step_count = 0
best_reward = -9999

agent.policy_net.load_state_dict(torch.load(agent.MODEL_FILE, weights_only=True))

for episode in range(agent.epoch):
    obs = env.reset()

    done = False
    episode_reward = 0.0
    max_episode_reward = 0.0

    state = preprocess(obs, IMAGE_CROP, agent.image_resize)

    # See frame after preprocessing
    # plt.figure()
    # plt.imshow(state[0], cmap="gray")
    # plt.title('Original Frame')
    # plt.show()

    while not done:
        env.render()
        action = agent.act(state)
        next_obs, rew, done, info = env.step(agent.actions[int(action)])
        episode_reward += rew
        next_state = preprocess(next_obs, IMAGE_CROP, agent.image_resize)

        # if episode_reward > max_episode_reward:
        #     max_episode_reward = episode_reward
        #
        # if step_count / 100 == 0 and episode_reward <= max_episode_reward:
        #     episode_reward -= 1

        if is_training:
            agent.replay_memory.append(state, action, rew, next_state, done)

        step_count += 1

        state = next_state

    print("Reward at end of episode:" + str(episode_reward))

    # Track rewards per episode
    rewards_per_episode.append(episode_reward)

    if is_training:
        if episode_reward > best_reward:
            log_message=f"New best reward:{episode_reward:0.1f}"
            print(log_message)
            with open(agent.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

            torch.save(agent.policy_net.state_dict(), agent.MODEL_FILE)
            best_reward = episode_reward

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
    epsilon_history.append(agent.epsilon)

    if (len(agent.replay_memory)) > agent.batch_size:
        batch = agent.replay_memory.sample()

        agent.optimize(batch)

        if step_count > agent.network_sync_rate:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            step_count = 0

env.close()