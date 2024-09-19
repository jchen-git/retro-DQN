# TODO
# Implement DQN to Gym Environment

import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Action Space = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

def main():
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)
        obs = env.reset()
        print(env.unwrapped.buttons)

        while True:
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                obs = env.reset()
                break
        env.close()


if __name__ == "__main__":
        main()
