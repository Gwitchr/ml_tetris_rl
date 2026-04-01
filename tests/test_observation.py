from src.env.tetris_env import TetrisEnv


def test_observation_shape_space()-> None:
    env = TetrisEnv()
    observation, _ = env.reset()

    assert observation.shape == (202,)
    assert env.observation_space.contains(observation)
