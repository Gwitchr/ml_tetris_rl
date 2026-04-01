from src.env.tetris_env import TetrisEnv


def test_reset_board() -> None:
    env = TetrisEnv()
    observation, info = env.reset()

    assert observation is not None
    assert env.board.shape == (20, 10 )
    assert env.board.sum() == 0.0
    assert 0 <= env.current_piece <= 6
    assert 0 <= env.next_piece <= 6
