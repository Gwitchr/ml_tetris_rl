from src.env.tetris_env import TetrisEnv


def main() -> None:
    env = TetrisEnv()

    observation,info = env.reset()

    print("Observation type: ", type(observation) )
    print("Observation shape: ", observation.shape )
    print("Observation dtype: ", observation.dtype )
    print("First 10 values: ", observation[:10] )
    print("Last 2 values (piece Ids): ", observation[-2:] )
    print("Info: ", info )

    print("Observation space: ", env.observation_space  )
    print("Action space: ", env.action_space  )

    print("Observation inside space: ", env.observation_space.contains(observation))

    env.render()

if __name__ == "__main__":
    main()
