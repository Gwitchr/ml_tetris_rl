from src.env.tetris_env import TetrisEnv


def test_terminated_top_row()-> None:
    env = TetrisEnv()
    env.reset()

    env.board[0,5] = 1.0

    _,_,terminated,_,_= env.step(0)

    assert terminated is True


def test_not_terminated_on_empty_board() -> None:
    env = TetrisEnv()
    env.reset()

    _,_,terminated,_,_= env.step(0)

    assert terminated is False
