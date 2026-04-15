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

def test_reward_survival_bonus()-> None:
    env = TetrisEnv()
    env.reset()

    _, reward, _, _, _ = env.step(0)

    assert reward == 0.1

def test_reward_with_line_clear()-> None:
    env = TetrisEnv()
    env.reset()

    env.board[19, 4:] = 1.0
    env.current_piece = 0

    _, reward, _,_,info = env.step(0)

    assert reward == 1.1
    assert info["lines_cleared_this_step"] == 1
