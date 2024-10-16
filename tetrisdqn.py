import os
import random
#import gym
import retro
import torch
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from fontTools.varLib import drop_implied_oncurve_points

from agentDQN import Agent
from preprocessing import preprocess, stack_frame

# TODO
# Change rewards given such that bottom four rows should be filled
# Create GUI

<<<<<<< Updated upstream
def create_graphs():
=======
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
    4:[[0,0,0,1],
       [4,5,6,6]],
    5:[[1,1,1,0],
       [4,5,6,4]],
    6:[[0,1,2,0],
       [5,5,5,6]],
    7:[[0,1,2,2],
       [5,5,5,4]],
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
    12:[[1,2,1,0],
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
        [4,5,6,3]],
    19:[[0,1,2,3],
        [5,5,5,5]],
    }

class Piece:
    def __init__(self, p_id):
        self.x, self.y = np.array(pieces[p_id])
        self.x_bounds = [4,6]
        self.y_bounds = [2,5]

        if p_id in [0,1,2,3]:
            self.rotations = [0,1,2,3]
        elif p_id in [4,5,6,7]:
            self.rotations = [4,5,6,7]
        elif p_id in [13,14,15,16]:
            self.rotations = [13,14,15,16]
        elif p_id in [8,9]:
            self.rotations = [8,9]
        elif p_id in [11,12]:
            self.rotations = [11,12]
        elif p_id in [17,18,19]:
            self.rotations = [18,19]
        elif p_id == 10:
            self.rotations = [10]

def get_info(curr_board):
    return [
        get_bumpiness(np.transpose(curr_board)),
        get_holes(np.transpose(curr_board)),
        get_agg_height(np.transpose(curr_board)),
        get_line_clears(curr_board)
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

def drop(piece, heights, col, moves, curr_board):
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
    states = []
    base_board = curr_board.copy()
    test_board = curr_board.copy()
    heights = get_col_heights(np.transpose(base_board))

    piece = Piece(p_id)

    for rotation_id in piece.rotations:
        rotations = []
        for i in range(abs(p_id-rotation_id)):
            if p_id < rotation_id:
                rotations.append([1,0,0,0,0,0,0,0,0])
            elif p_id > rotation_id:
                rotations.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

        if rotation_id in [6, 9, 12, 13, 17, 18, 19]:
            left_moves = 5
        elif rotation_id in [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 15, 16]:
            left_moves = 4
        elif rotation_id == 18:
            left_moves = 3
        else:
            left_moves = 0

        for col in range(0,6):
            piece = Piece(rotation_id)
            place_piece(piece, test_board)
            if drop(piece, heights, col, left_moves, test_board):
                actions_list=[]
                for rotate in rotations:
                    actions_list.append(rotate)
                for i in range(left_moves):
                    actions_list.append([0,0,0,0,0,0,1,0,0])
                actions_list.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
                states.append((actions_list, get_info(test_board)))
            test_board = base_board.copy()
            left_moves -= 1

        if rotation_id in [3, 4, 10, 15, 17]:
            right_moves = 4
        elif rotation_id in [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 18]:
            right_moves = 3
        else:
            right_moves = 0

        for col in range(9,5,-1):
            piece = Piece(rotation_id)
            place_piece(piece, test_board)
            if drop(piece, heights, col, -right_moves, test_board):
                actions_list=[]
                for rotate in rotations:
                    actions_list.append(rotate)
                for i in range(right_moves):
                    actions_list.append([0,0,0,0,0,0,0,1,0])
                actions_list.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
                states.append((actions_list, get_info(test_board)))
            test_board = base_board.copy()
            right_moves -= 1

    if test_board[0][5] == 1:
        states.append(([[0, 0, 0, 0, 0, 1, 0, 0, 0]],get_info(test_board)))

    if len(states) == 0:
        states.append(([[0, 0, 0, 0, 0, 1, 0, 0, 0]],get_info(test_board)))

    return states

def get_line_clears(curr_board):
    line_clears = 0
    for i in range(len(curr_board)):
        if (curr_board[i] == 1).all():
            line_clears += 1
    return line_clears


def get_bumpiness(curr_board):
    bumpiness_num = 0
    current_height = 0
    last_height = -1
    for i in range(len(curr_board)):
        if curr_board[i].any():
            current_height = 20 - np.where(curr_board[i] == 1)[0][0]
        if last_height != -1:
            bumpiness_num += abs(last_height - current_height)
        last_height = current_height
    return bumpiness_num

def get_col_heights(curr_board):
    heights = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(curr_board)):
        if curr_board[i].any():
            heights[i] = 20 - np.where(curr_board[i] == 1)[0][0]
    return heights

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

def create_graph(name, variable):
>>>>>>> Stashed changes
    # Save plots
    fig = plt.figure(1)
    durations_t = torch.tensor(variable, dtype=torch.float)
    # Plot average rewards (Y-axis) vs episodes (X-axis)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
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
<<<<<<< Updated upstream
env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)
num_actions = env.action_space.n
IMAGE_CROP = (35, 204, 85, 170)
INPUT_SHAPE = (4, 128, 128)
=======
env = retro.make("Tetris-Nes", state=SAVE_STATES[1], inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM)
IMAGE_CROP = (35, 204, 85, 170)
#INPUT_SHAPE = (4, 84, 84)
INPUT_SHAPE = 4
>>>>>>> Stashed changes
agent = Agent(INPUT_SHAPE,"tetris", training=True)

is_training = True
rewards_per_episode = []
score_per_episode = []
holes_per_episode = []
agg_height_per_ep = []
line_clears_per_ep = []
bump_per_episode = []
epsilon_history = []
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
<<<<<<< Updated upstream
    env.load_state(random.choice(SAVE_STATES))
    state = env.reset()

    done = False
    episode_reward = 0.0

    frame = preprocess(state, IMAGE_CROP, agent.image_resize)
    frames = stack_frame(None, frame, True)
    state = torch.tensor(frames, dtype=torch.float32, device="cuda").unsqueeze(0)
=======
    #env.load_state(random.choice(SAVE_STATES))
    obs = env.reset()
    board = np.array(obs[0x0400:0x04C8].reshape((20, 10)))
    board[board == 239] = 0
    board[board != 0] = 1
    piece_id = obs[0x0042]
    prev_state = get_info(board)
    prev_state = torch.tensor([prev_state], device=device, dtype=torch.float)

    ep_bump = 0.0
    ep_agg_height = 0.0
    ep_holes = 0.0
    ep_line_clears = 0.0
    ep_reward = 0.0
    done = False
    can_move = True

    # frame = preprocess(env.render('rgb_array'), IMAGE_CROP, agent.image_resize)
    # frames = stack_frame(None, frame, True)
    # state = torch.tensor(frames, dtype=torch.float32, device="cuda").unsqueeze(0)
>>>>>>> Stashed changes

    while not done:
        current_step = 0
        total_reward = 0
        env.render()

<<<<<<< Updated upstream
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
=======
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
            can_move = False

        if int(info['game_phase'] == 6 and not can_move):
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
    ep_bump, ep_agg_height, ep_holes, ___ = get_info(board)
    bump_per_episode.append(ep_bump)
    agg_height_per_ep.append(ep_agg_height)
    holes_per_episode.append(ep_holes)
    line_clears_per_ep.append(ep_line_clears)
    rewards_per_episode.append(ep_reward)
>>>>>>> Stashed changes
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
            create_graph('rewards', rewards_per_episode)
            create_graph('bumpiness', bump_per_episode)
            create_graph('agg_height', agg_height_per_ep)
            create_graph('holes', holes_per_episode)
            create_graph('epsilon', epsilon_history)

env.close()

log_message=f"{datetime.now()}: Training Complete"
print(log_message)
with open(agent.LOG_FILE, 'a') as file:
    file.write(log_message + '\n')