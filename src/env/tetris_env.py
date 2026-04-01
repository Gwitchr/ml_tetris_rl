import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

        # Observation
        obs_size = self.board_height * self.board_width + 2

        self.observation_space = spaces.Box(
            low=0.0,
            high=6.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.board_width)

    def _get_observation(self) -> np.ndarray:
        board_flat = self.board.flatten()
        piece_info = np.array([self.current_piece, self.next_piece], dtype=np.float32)
        observation = np.concatenate([board_flat, piece_info])
        return observation

    def get_info(self) -> dict:
        return {"current": self.current_piece, "next": self.next_piece}

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.board = np.zeros((self.board_height, self.board_width), dtype=np.float32)

        self.current_piece = int(self.np_random.integers(0, self.num_pieces_types))
        self.next_piece = int(self.np_random.integers(0, self.num_pieces_types))

        observation = self._get_observation()
        info = self.get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._drop_piece(action)
        reward = 1.0
        terminated = np.any(self.board[0] == 1.0)
        truncated = False

        self.current_piece = self.next_piece
        self.next_piece = int(self.np_random.integers(0, self.num_pieces_types))

        observation = self._get_observation()
        info = self.get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        print("Board shape:", self.board.shape)
        print("Current piece:", self.current_piece)
        print("Next piece:", self.next_piece)

    def _drop_piece(self, column: int) -> None:
        """
        Drop a single block into a column
        Fill the lowest available cell
        """
        for row in reversed(range(self.board_height)):
            print(
                f"Checking cell row={row}, col={column}, value={self.board[row, column]}"
            )
            if self.board[row, column] == 0:
                print(f"Placing block at row:{row}, col:{column}")
                self.board[row, column] = 1.0
                return
