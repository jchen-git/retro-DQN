import os
import gym_tetris
import torch
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from agentDQN import Agent

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##
# Used to get all possible end states of the current game state and get features from the end states
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
        [3,4,5,6]],
    19:[[0,0,0,0],
        [3,4,5,6]]
    }

class Piece:
    def __init__(self, p_id):
        self.x, self.y = np.array(pieces[p_id])

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
        elif p_id in [17,18,19]:
            self.rotations = [18,17]
        elif p_id == 10:
            self.rotations = [10]

def get_info(curr_board):
    line_clears, line_cleared_board = get_line_clears(curr_board)
    return [
        line_clears,
        get_bumpiness(np.transpose(line_cleared_board)),
        get_agg_height(np.transpose(line_cleared_board)),
        get_holes(np.transpose(line_cleared_board)),
        get_high_low(np.transpose(line_cleared_board))
    ]

def check_valid(piece, curr_board):
    if (piece.x < 0).sum() > 0 or (piece.x > 19).sum() > 0:
        piece.x -= 1
        return False
    for i in range(4):
        if curr_board[piece.x[i]][piece.y[i]] == 1:
            piece.x -= 1
            return False
    return True

def drop(piece, moves, curr_board):
    piece.y -= moves

    if (piece.y < 0).sum() > 0 or 10 in piece.y:
        return False

    while check_valid(piece, curr_board):
        piece.x += 1

    if (piece.x == 1).sum() > 0:
        return False

    for i in range(4):
        if curr_board[piece.x[i]][piece.y[i]] == 1:
            return False
        else:
            curr_board[piece.x[i]][piece.y[i]] = 1
    return True

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
            rotations.append(1)
        rotation_num += 1

        # Check available left input end states
        if rotation_id in [1, 6, 9, 12, 13, 17]:
            left_moves = 5
        elif rotation_id in [0, 2, 3, 4, 5, 7, 8, 10, 11, 14, 15, 16]:
            left_moves = 4
        elif rotation_id == 18:
            left_moves = 3
        else:
            left_moves = 0

        while left_moves >= 0:
            piece = Piece(rotation_id)
            if drop(piece, left_moves, test_board):
                actions_list=[]
                for rotate in rotations:
                    actions_list.append(rotate)
                    actions_list.append(0)
                for i in range(left_moves):
                    actions_list.append(4)
                    actions_list.append(0)
                action_states.append((actions_list, get_info(test_board)))
            test_board = base_board.copy()
            left_moves -= 1

        # Check available right input end states
        if rotation_id in [3, 4, 10, 15, 17]:
            right_moves = 4
        elif rotation_id in [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 18]:
            right_moves = 3
        else:
            right_moves = 0

        while right_moves >= 0:
            piece = Piece(rotation_id)
            if drop(piece, -right_moves, test_board):
                actions_list=[]
                for rotate in rotations:
                    actions_list.append(rotate)
                    actions_list.append(0)
                for i in range(right_moves):
                    actions_list.append(3)
                    actions_list.append(0)
                action_states.append((actions_list, get_info(test_board)))
            test_board = base_board.copy()
            right_moves -= 1

    if len(action_states) == 0:
        action_states.append(([0], get_info(test_board)))

    return action_states

def get_line_clears(curr_board):
    line_clears = 0
    for i in range(len(curr_board)):
        if (curr_board[i] == 1).all():
            line_clears += 1
            curr_board[i] = [0]
            for j in range((i - 1), 0, -1):
                curr_board[j + 1] = curr_board[j]
                if all(curr_board[j] == 0):
                    break
    return line_clears, curr_board

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
##

def create_graph(directory, model_name, metric, variable, x_label, y_label):
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
    fig.savefig(os.path.join(directory, f'{model_name}_{metric}.png'))
    plt.close(fig)

# Set path for custom ROMs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

matplotlib.use('Agg')

def run(render_game, ep_time, ai_model, model_dir, log_dir, gui_episodes, gui_child_done_conn, gui_child_conn, use_timer):
    # Tetris game
    env = gym_tetris.make("TetrisA-v3")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    INPUT_SHAPE = 5
    is_training = True
    model_file = os.path.join(model_dir, f'{ai_model}')
    log_file = os.path.join(log_dir, f'{ai_model}.log')
    data_file = os.path.join(log_dir, f'{ai_model}.dat')
    agent = Agent(INPUT_SHAPE,"tetris", is_training)

    # Logging variables
    rewards_per_episode = []
    score_per_episode = []
    avg_holes_per_episode = []
    avg_bump_per_episode = []
    line_clears_per_ep = []
    epsilon_history = []

    best_reward = -999.0
    timestep = 0

    log_message=f"{datetime.now().strftime('%H:%M:%S')}: Training..."
    print(log_message)
    with open(log_file, 'w') as file:
        file.write(log_message + '\n')

    if os.path.isfile(data_file):
        with open(data_file, 'r') as file:
            best_reward = float(file.read())
    else:
        with open(data_file, 'w') as file:
            file.write(str(best_reward))

    if os.path.isfile(model_file):
        if os.path.getsize(model_file) != 0:
            agent.policy_net.load_state_dict(torch.load(model_file, weights_only=True, map_location=device))

    for episode in range(agent.episodes):
        if use_timer:
            hard_stop = time.time() + ep_time
        ep_start_time = time.time()
        env.reset()
        obs = env.ram
        board = np.array(obs[0x0400:0x04C8].reshape((20, 10)))
        board[board == 239] = 0
        board[board != 0] = 1
        piece_id = obs[0x0042]
        prev_state = get_info(board)
        prev_state = torch.tensor([prev_state], device=device, dtype=torch.float)

        ep_bump = []
        ep_holes = []
        ep_line_clears = []
        ep_reward = 0.0
        done = False
        can_move = True

        while not done:
            ___, rew, done, info = env.step(5)

            if render_game:
                env.render()
            obs = env.ram

            if not done:
                rew += 1
            ep_reward += rew

            # 0x0041 in RAM refers to the current Tetris piece y position
            if obs[0x0041] == 0 and can_move:
                board = np.array(obs[0x0400:0x04C8].reshape((20, 10)))
                board[board == 239] = 0
                board[board != 0] = 1
                piece_id = obs[0x0042]
                actions, state = agent.act(get_possible_states(piece_id, board))
                for action in actions:
                    if not done:
                        ___, rew, done, info = env.step(action)
                        if render_game:
                            env.render()
                        ep_reward += rew
                if not done:
                    ___, rew, done, info = env.step(0)
                if render_game:
                    env.render()
                obs = env.ram

                ep_bump.append(state[1])
                ep_holes.append(state[3])

                can_move = False

            # 0x0048 refers to the current Tetris game phase. > 5 is during the score counter update
            if obs[0x0048] > 5 and not can_move:
                rew += state[0] * agent.line_clear_weight
                rew += state[1] * agent.bumpiness_weight
                rew += state[2] * agent.agg_height_weight
                rew += state[3] * agent.hole_weight

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

            if use_timer:
                if time.time() > hard_stop:
                    done = True

        gui_episodes.value = episode

        # Logging variables to save to an image
        avg_bump_per_episode.append(sum(ep_bump) / len(ep_bump))
        avg_holes_per_episode.append(sum(ep_holes) / len(ep_holes))
        line_clears_per_ep.append(ep_line_clears)
        rewards_per_episode.append(ep_reward)
        score_per_episode.append(int(info['score']))
        epsilon_history.append(agent.epsilon)

        # Saves model if the current episode's reward is the highest in the current training session
        if ep_reward > best_reward:
            log_message= f"{datetime.now().strftime('%H:%M:%S')}: New best reward: {ep_reward:0.2f} at episode {episode}"
            gui_child_conn.send(log_message)
            print(log_message)

            with open(log_file, 'a') as file:
                file.write(log_message + '\n')

            torch.save(agent.policy_net.state_dict(), model_file)
            best_reward = ep_reward

            with open(data_file, 'w') as file:
                file.write(str(best_reward))


        log_message = f"{datetime.now().strftime('%H:%M:%S')}: {episode}/{agent.episodes} complete [{time.time() - ep_start_time}/episode]"
        gui_child_conn.send(log_message)
        print(log_message)
        create_graph(log_dir, ai_model,'rewards', rewards_per_episode, 'episode', 'reward')
        # Graphs used for testing
        #create_graph(log_dir, ai_model, 'bumpiness', avg_bump_per_episode, 'episode', 'avg. bumpiness')
        #create_graph(log_dir, ai_model,'holes', avg_holes_per_episode, 'episode', 'avg. holes')
        #create_graph(log_dir, ai_model,'epsilon', epsilon_history, 'episode', 'epsilon')
        #create_graph(log_dir, ai_model, 'score', score_per_episode, 'episode', 'score')

    env.close()

    gui_child_done_conn.send(True)

    log_message=f"{datetime.now()}: Training Complete"
    print(log_message)
    with open(log_file, 'a') as file:
        file.write(log_message + '\n')