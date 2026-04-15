import numpy as np

from src.env.tetris_env import TETROMINOES, TetrisEnv


def test_i_piece_fits_column_0()-> None:
    env = TetrisEnv()
    env.reset()

    env.current_piece = 0
    env._place_piece(0)

    assert env.board[19,0:4].sum() == 4.0


def test_i_piece_edge() -> None:
    env = TetrisEnv()
    env.reset()

    env.current_piece = 0
    env._place_piece(9)

    assert env.board[19, 6:10].sum() == 4.0
