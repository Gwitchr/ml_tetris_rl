from src.env.tetris_env import TetrisEnv


def test_clear_full_lines_removes_row() -> None:
    env = TetrisEnv()
    env.reset()

    env.board[19] = 1.0
    env.board[18, 0] = 1.0

    cleared = env._clear_full_lines()

    assert cleared == 1
    assert env.board[19,0] == 1.0
    assert env.board[19].sum() == 1.0
    assert env.board[0].sum() == 0.0
