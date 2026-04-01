import argparse

from src.env.tetris_env import TetrisEnv


def main(debug: bool = False) -> None:
    env = TetrisEnv()

    observation, info = env.reset()

    print("Starting env test...")

    for step in range(10):
        action = env.action_space.sample()

        print(f"\nStep {step}")
        print(f"Action (column): {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Top row:", env.board[0])
        print(env.board)

        if terminated:
            print("Game over")
            break

    if debug:
        print("Observation type: ", type(observation))
        print("Observation shape: ", observation.shape)
        print("Observation dtype: ", observation.dtype)
        print("First 10 values: ", observation[:10])
        print("Last 2 values (piece Ids): ", observation[-2:])
        print("Info: ", info)

        print("Observation space: ", env.observation_space)
        print("Action space: ", env.action_space)

        print("Observation inside space: ", env.observation_space.contains(observation))

        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(debug=args.debug)
