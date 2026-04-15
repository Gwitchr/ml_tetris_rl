import gymnasium as gym
import numpy as np
from gymnasium import spaces

TETROMINOES = {
    0: np.array([[1, 1, 1, 1]]),  # I
    1: np.array([[1, 1], [1, 1]]),  # O
    2: np.array([[0, 1, 0], [1, 1, 1]]),  # T
    3: np.array([[1, 0, 0], [1, 1, 1]]),  # L
    4: np.array([[0, 0, 1], [1, 1, 1]]),  # J
    5: np.array([[1, 1, 0], [0, 1, 1]]),  # S
    6: np.array([[0, 1, 1], [1, 1, 0]]),  # Z
}


class TetrisEnv(gym.Env):
    """
    Minimal Tetris environment skeleton.

    Observation:
    - board: 20x10 grid of 0s and 1s
    - current_piece: integer in [0, 6]
    - next_piece: integer in [0, 6]

    Returned observation:
    - flattened NumPy array of shape (202,)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self) -> None:
        super().__init__()

        self.board_height = 20
        self.board_width = 10
        self.num_pieces_types = 7

        # Internal state
        self.board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
        self.current_piece = 0
        self.next_piece = 0
        self.score = 0.0
        self.lines_cleared_total = 0

        # Observation
        obs_size = self.board_height * self.board_width + 2

        self.observation_space = spaces.Box(
            low=0.0,
            high=6.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        # temporal to avoid edge issues
        self.action_space = spaces.Discrete(self.board_width - 3)

    def _get_observation(self) -> np.ndarray:
        board_flat = self.board.flatten()
        piece_info = np.array([self.current_piece, self.next_piece], dtype=np.float32)
        observation = np.concatenate([board_flat, piece_info])
        return observation

    def get_info(self) -> dict:
        return {
            "current": int(self.current_piece),
            "next": int(self.next_piece),
            "score": float(self.score),
            "lines_cleared": int(self.lines_cleared_total),
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
        self.score = 0.0
        self.lines_cleared_total = 0

        self.current_piece = int(self.np_random.integers(0, self.num_pieces_types))
        self.next_piece = int(self.np_random.integers(0, self.num_pieces_types))

        observation = self._get_observation()
        info = self.get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._place_piece(action)

        lines_cleared = self._clear_full_lines()
        self.lines_cleared_total += lines_cleared

        # minimal reward
        reward = float(lines_cleared)

        # survival reward
        reward += 0.1

        self.score += reward

        terminated = bool(np.any(self.board[0] == 1.0))
        truncated = False

        self.current_piece = self.next_piece
        self.next_piece = int(self.np_random.integers(0, self.num_pieces_types))

        observation = self._get_observation()
        info = self.get_info()
        info["lines_cleared_this_step"] = lines_cleared

        return observation, reward, terminated, truncated, info

    def render(self):
        print("Board shape:", self.board.shape)
        print("Current piece:", self.current_piece)
        print("Next piece:", self.next_piece)

    def _place_piece(self, column: int) -> None:
        piece = TETROMINOES[self.current_piece]

        piece_height, piece_width = piece.shape

        # Start from bottom and move upward
        for row in reversed(range(self.board_height - piece_height + 1)):
            fits = True

            # fits?
            for row_height in range(piece_height):
                for col_width in range(piece_width):
                    if piece[row_height, col_width] == 1:
                        if self.board[row + row_height, column + col_width] == 1:
                            fits = False
                            break
                if not fits:
                    break

            if fits:
                for row_height in range(piece_height):
                    for col_width in range(piece_width):
                        if piece[row_height, col_width] == 1:
                            self.board[row + row_height, column + col_width] = 1.0
                return

    def _clear_full_lines(self) -> int:
        """
        Clear filled rows and return how many were cleared
        """
        full_rows = []

        for row in range(self.board_height):
            if np.all(self.board[row] == 1.0):
                full_rows.append(row)

        num_cleared = len(full_rows)

        if num_cleared == 0:
            return 0

        # keep other rows
        remaining_rows = []

        for row in range(self.board_height):
            if row not in full_rows:
                remaining_rows.append(self.board[row].copy())

        # new empty rows
        new_rows = [
            np.zeros((self.board_width,), dtype=np.float32) for _ in range(num_cleared)
        ]

        # stack rows
        self.board = np.vstack(new_rows + remaining_rows)

        return num_cleared
