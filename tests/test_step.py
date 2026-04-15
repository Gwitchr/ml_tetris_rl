from src.env.tetris_env import TetrisEnv


def test_step_returns_expected_types() -> None:
    env = TetrisEnv()
    env.reset()

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == (202, )
    assert reward is not None
    assert isinstance(reward, float)
    assert isinstance(terminated,bool)
    assert isinstance(truncated,bool)
    assert isinstance(info,dict)

def test_step_changes_board() -> None:
    env = TetrisEnv()
    env.reset()

    board_before = env.board.copy()

    env.step(0)

    assert not (env.board == board_before).all()
