import os
import random
#import gym
import retro
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from agentDQN import Agent
from preprocessing import preprocess, stack_frame

# TODO
# Change rewards given such that bottom four rows should be filled
# Create GUI
# Check why flat I piece (id 18) desyncs the state

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pieces = {
    # T
    0:[[1,1,1,0],
       [4,5,6,5]],
    1:[[0,1,2,1],
       [5,5,5,6]],
    2:[[0,0,0,1],
       [4,5,6,5]],
    3:[[0,1,2,1],
       [5,5,5,4]],
    # J
    4:[[0,1,2,2],
       [5,5,5,4]],
    5:[[1,1,1,0],
       [4,5,6,4]],
    6:[[0,1,2,0],
       [5,5,5,6]],
    7:[[0,0,0,1],
       [4,5,6,6]],
    # Z
    8:[[0,0,1,1],
       [4,5,5,6]],
    9:[[2,1,1,0],
       [5,5,6,6]],
    # O
    10:[[0,0,1,1],
        [4,5,4,5]],
    # S
    11:[[1,1,0,0],
        [4,5,5,6]],
    12:[[0,1,1,2],
        [5,5,6,6]],
    # L
    13:[[0,1,2,2],
        [5,5,5,6]],
    14:[[0,1,0,0],
        [4,4,5,6]],
    15:[[0,1,2,0],
        [5,5,5,4]],
    16:[[1,1,1,0],
        [4,5,6,6]],
    # I
    17:[[0,1,2,3],
        [5,5,5,5]],
    18:[[0,0,0,0],
        [3,4,5,6]]
    }

class Piece:
    def __init__(self, p_id):
        self.x, self.y = np.array(pieces[p_id])
        self.x_bounds = [4,6]
        self.y_bounds = [2,5]

        if p_id in [0,1,2,3]:
            self.rotations = [2,3,0,1]
        elif p_id in [4,5,6,7]:
            self.rotations = [7,4,5,6]
        elif p_id in [13,14,15,16]:
            self.rotations = [14,15,16,13]
        elif p_id in [8,9]:
            self.rotations = [8,9]
        elif p_id in [11,12]:
            self.rotations = [11,12]
        elif p_id in [17,18]:
            self.rotations = [18,17]
        elif p_id == 10:
            self.rotations = [10]

def get_info(curr_board):
    return [
        get_line_clears(curr_board),
        get_bumpiness(np.transpose(curr_board)),
        get_agg_height(np.transpose(curr_board)),
        get_holes(np.transpose(curr_board)),
        get_high_low(np.transpose(curr_board))
    ]

def check_valid(piece, curr_board):
    if (piece.x < 0).sum() > 0 or 20 in piece.x:
        piece.x -= 1
        return False
    for i in range(4):
        if curr_board[piece.x[i]][piece.y[i]] == 1:
            piece.x -= 1
            return False
    return True

def drop(piece, moves, curr_board):
    for i in range(4):
        curr_board[piece.x[i]][piece.y[i]] = 0

    piece.y -= moves

    if (piece.y < 0).sum() > 0 or 10 in piece.y:
        return False

    while check_valid(piece, curr_board):
        piece.x += 1
        if not check_valid(piece, curr_board):
            break

    for i in range(4):
        if curr_board[piece.x[i]][piece.y[i]] == 1:
            return False
        else:
            curr_board[piece.x[i]][piece.y[i]] = 1
    return True

def place_piece(piece, curr_board):
    for i in range(4):
        curr_board[piece.x[i]][piece.y[i]] = 1
    return curr_board

def get_possible_states(p_id, curr_board):
    # List of tuple containing (action, board state)
    action_states = []
    base_board = curr_board.copy()
    test_board = curr_board.copy()

    piece = Piece(p_id)

    rotation_num = 0
    for rotation_id in piece.rotations:
        rotations = []
        for i in range(rotation_num):
            rotations.append([0,0,0,0,0,0,0,0,1])
        rotation_num += 1

        # Double check
        if rotation_id in [1, 6, 9, 12, 13, 17]:
            left_moves = 5
        elif rotation_id in [0, 2, 3, 4, 5, 7, 8, 10, 11, 14, 15, 16]:
            left_moves = 4
        elif rotation_id == 18:
            left_moves = 3
        else:
            left_moves = 0

        for col in range(0,5):
            piece = Piece(rotation_id)
            place_piece(piece, test_board)
            if drop(piece, left_moves, test_board):
                actions_list=[]
                for rotate in rotations:
                    actions_list.append(rotate)
                for i in range(left_moves):
                    actions_list.append([0,0,0,0,0,0,1,0,0])
                actions_list.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
                action_states.append((actions_list, get_info(test_board)))
            test_board = base_board.copy()
            left_moves -= 1

        # Double check
        if rotation_id in [3, 4, 10, 15, 17]:
            right_moves = 4
        elif rotation_id in [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 18]:
            right_moves = 3
        else:
            right_moves = 0

        for col in range(9,5,-1):
            piece = Piece(rotation_id)
            place_piece(piece, test_board)
            if drop(piece, -right_moves, test_board):
                actions_list=[]
                for rotate in rotations:
                    actions_list.append(rotate)
                for i in range(right_moves):
                    actions_list.append([0,0,0,0,0,0,0,1,0])
                actions_list.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
                action_states.append((actions_list, get_info(test_board)))
            test_board = base_board.copy()
            right_moves -= 1

    if len(action_states) == 0:
        action_states.append(([[0, 0, 0, 0, 0, 1, 0, 0, 0]],get_info(test_board)))

    return action_states

def get_line_clears(curr_board):
    line_clears = 0
    for i in range(len(curr_board)):
        if (curr_board[i] == 1).all():
            line_clears += 1
    return line_clears


def get_bumpiness(curr_board):
    bumpiness_num = 0
    last_height = -1
    for i in range(len(curr_board)):
        if curr_board[i].any():
            current_height = 20 - np.where(curr_board[i] == 1)[0][0]
        else:
            current_height = 0
        if last_height != -1:
            bumpiness_num += abs(last_height - current_height)
        last_height = current_height
    return bumpiness_num

# def get_col_heights(curr_board):
#     heights = [0,0,0,0,0,0,0,0,0,0]
#     for i in range(len(curr_board)):
#         if curr_board[i].any():
#             heights[i] = 20 - np.where(curr_board[i] == 1)[0][0]
#     return heights

def get_agg_height(curr_board):
    height = 0
    for i in range(len(curr_board)):
        if curr_board[i].any():
            height += 20 - np.where(curr_board[i] == 1)[0][0]
    return height

def get_holes(curr_board):
    num_holes = 0
    for i in range(len(curr_board)):
        if curr_board[i].any():
            for j in range(np.where(curr_board[i] == 1)[0][0], len(curr_board[i])):
                if curr_board[i][j] == 0:
                    num_holes += 1
    return num_holes

def get_high_low(curr_board):
    max_h = 0
    min_h = 20
    for i in range(len(curr_board)):
        if curr_board[i].any():
            h = 20 - np.where(curr_board[i] == 1)[0][0]
        else:
            h = 0

        if h > max_h:
            max_h = h
        if h < min_h:
            min_h = h
    return max_h - min_h

def create_graph(name, variable, x_label, y_label):
    # Save plots
    fig = plt.figure(1)
    durations_t = torch.tensor(variable, dtype=torch.float)
    # Plot average rewards (Y-axis) vs episodes (X-axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Save plots
    fig.savefig(os.path.join(agent.GRAPH_FILE, f'_{name}.png'))
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
env = retro.make("Tetris-Nes", state=SAVE_STATES[0], inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM)
IMAGE_CROP = (35, 204, 85, 170)
#INPUT_SHAPE = (4, 84, 84)
INPUT_SHAPE = 5
agent = Agent(INPUT_SHAPE,"tetris", training=True)

is_training = True
rewards_per_episode = []
score_per_episode = []
avg_holes_per_episode = []
avg_agg_height_per_ep = []
avg_bump_per_episode = []
line_clears_per_ep = []
epsilon_history = []
best_reward = -999.0
timestep = 0

log_message=f"{datetime.now()}: Training..."
print(log_message)
with open(agent.LOG_FILE, 'w') as file:
    file.write(log_message + '\n')

if os.path.isfile(agent.DATA_FILE):
    with open(agent.DATA_FILE, 'r') as file:
        best_reward = float(file.read())
else:
    with open(agent.DATA_FILE, 'w') as file:
        file.write(str(best_reward))

if os.path.isfile(agent.MODEL_FILE):
    agent.policy_net.load_state_dict(torch.load(agent.MODEL_FILE, weights_only=True))

for episode in range(agent.epoch):
    #env.load_state(random.choice(SAVE_STATES))
    obs = env.reset()
    board = np.array(obs[0x0400:0x04C8].reshape((20, 10)))
    board[board == 239] = 0
    board[board != 0] = 1
    piece_id = obs[0x0042]
    prev_state = get_info(board)
    prev_state = torch.tensor([prev_state], device=device, dtype=torch.float)

    ep_bump = []
    ep_agg_height = []
    ep_holes = []
    ep_line_clears = []
    ep_reward = 0.0
    done = False
    can_move = True

    while not done:
        current_step = 0
        total_reward = 0
        env.render()

        obs, rew, done, info = env.step([0, 0, 0, 0, 0, 1, 0, 0, 0])

        if not done:
            rew += 1
        ep_reward += rew

        if int(info['piece_y_pos']) == 1 and can_move:
            board = np.array(obs[0x0400:0x04C8].reshape((20, 10)))
            board[board == 239] = 0
            board[board != 0] = 1
            piece_id = obs[0x0042]
            actions, state = agent.act(get_possible_states(piece_id, board))
            for action in actions:
                obs, rew, done, info = env.step(action)
                obs, rew, done, info = env.step([0,1,0,0,0,0,0,0,0])
                ep_reward += rew

            ep_bump.append(state[1])
            ep_agg_height.append(state[2])
            ep_holes.append(state[3])

            can_move = False

        if int(info['game_phase'] == 6 and not can_move):
            board = np.array(obs[0x0400:0x04C8].reshape((20, 10)))
            board[board == 239] = 0
            board[board != 0] = 1
            piece_id = obs[0x0042]

            rew -= state[3] * 0.035

            if get_info(board) != state:
                print()

            actions = torch.tensor(actions, device=device, dtype=torch.int64)
            state = torch.tensor([state], device=device, dtype=torch.float)
            rew = torch.tensor([rew], device=device, dtype=torch.float)
            done = torch.tensor(done, device=device, dtype=torch.float)
            agent.replay_memory.append(prev_state, actions, rew, state, done)
            prev_state = state
            can_move = True

        timestep += 1

        if timestep > agent.update_rate:
            if (len(agent.replay_memory)) > agent.mini_batch_size:
                agent.optimize()
            timestep = 0


    # Logs
    avg_bump_per_episode.append(sum(ep_bump) / len(ep_bump))
    avg_agg_height_per_ep.append(sum(ep_agg_height) / len(ep_agg_height))
    avg_holes_per_episode.append(sum(ep_holes) / len(ep_holes))
    line_clears_per_ep.append(ep_line_clears)
    rewards_per_episode.append(ep_reward)
    score_per_episode.append(int(info['score']))
    epsilon_history.append(agent.epsilon)

    if is_training:
        if ep_reward > best_reward:
            log_message= f"{datetime.now()}: New best reward: {ep_reward:0.2f} at episode {episode}"
            print(log_message)
            with open(agent.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

            torch.save(agent.policy_net.state_dict(), agent.MODEL_FILE)
            best_reward = ep_reward

            with open(agent.DATA_FILE, 'w') as file:
                file.write(str(best_reward))

        if episode % 50 == 0:
            log_message = f"{datetime.now()}: Episode {episode} complete"
            print(log_message)
            create_graph('rewards', rewards_per_episode, 'episode', 'reward')
            create_graph('bumpiness', avg_bump_per_episode, 'episode', 'avg. bumpiness')
            create_graph('agg_height', avg_agg_height_per_ep, 'episode', 'avg. agg. height')
            create_graph('holes', avg_holes_per_episode, 'episode', 'avg. holes')
            create_graph('epsilon', epsilon_history, 'episode', 'epsilon')

env.close()

log_message=f"{datetime.now()}: Training Complete"
print(log_message)
with open(agent.LOG_FILE, 'a') as file:
    file.write(log_message + '\n')