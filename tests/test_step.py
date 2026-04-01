from src.env.tetris_env import TetrisEnv


def test_step_changes_board() -> None:
    env = TetrisEnv()
    env.reset()

    board_before = env.board.copy()

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == (202, )
    assert reward is not None
    assert isinstance(terminated,bool)
    assert isinstance(truncated,bool)
    assert isinstance(info,dict)
    assert not (env.board == board_before).all()
