"""
KinDER benchmark data source for HDF5-based robot demonstration datasets.

Reference: https://huggingface.co/datasets/kinder-bench/kinder-datasets

HDF5 schema:
    data/
      demo_0/
        actions       (T, action_dim)
        obs/
          robot_state (T, robot_dim)
          env_state   (T, env_dim)
          image       (T, 224, 224, 3)   -- 2D environments
          base_image  (T, 224, 224, 3)   -- 3D environments
          wrist_image (T, 224, 224, 3)   -- 3D environments
          overview_image (T, 224, 224, 3) -- 3D environments
      demo_1/
        ...
"""

from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from einops import rearrange

from ..base import DataSource, TrajectoryData


class KinDERDataSource(DataSource):
    """
    Data source for KinDER benchmark robot demonstration datasets.

    Loads actions and states into RAM at init; reads images from HDF5
    on-demand in load_visual_frames (WorldModelDataset handles resizing).

    Args:
        data_path: Path to the .hdf5 file.
        image_key: Which camera to use. "image" for 2D envs;
                   "base_image" / "wrist_image" / "overview_image" for 3D envs.
        state_keys: List of obs/* keys to concatenate into the state vector.
                    Defaults to ["robot_state"].
        n_rollout: Limit to the first N demos (None = all).
    """

    def __init__(
        self,
        data_path: str,
        image_key: str = "image",
        state_keys: Optional[List[str]] = None,
        n_rollout: Optional[int] = None,
        **kwargs,
    ):
        if state_keys is None:
            state_keys = ["robot_state"]
        # OmegaConf may pass a ListConfig; convert to a plain list for h5py compatibility
        state_keys = list(state_keys)

        self.hdf5_path = Path(data_path)
        self.image_key = image_key
        self.state_keys = state_keys

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"KinDER HDF5 file not found: {self.hdf5_path}")

        print(f"Loading KinDER trajectories from {self.hdf5_path}...")

        self._actions: List[torch.Tensor] = []
        self._states: List[torch.Tensor] = []
        self._seq_lengths: List[int] = []
        self._demo_keys: List[str] = []

        with h5py.File(self.hdf5_path, "r") as f:
            all_keys = sorted(f["data"].keys())
            if n_rollout is not None:
                all_keys = all_keys[:n_rollout]

            for key in all_keys:
                ep = f["data"][key]
                T = ep["actions"].shape[0]

                actions = torch.from_numpy(ep["actions"][:]).float()

                state_parts = []
                for sk in state_keys:
                    if sk in ep["obs"]:
                        state_parts.append(ep["obs"][sk][:])
                    else:
                        raise KeyError(
                            f"State key '{sk}' not found in {self.hdf5_path}[data/{key}/obs]. "
                            f"Available: {list(ep['obs'].keys())}"
                        )
                state_np = np.concatenate(state_parts, axis=-1)
                states = torch.from_numpy(state_np).float()

                self._actions.append(actions)
                self._states.append(states)
                self._seq_lengths.append(T)
                self._demo_keys.append(key)

        self.num_trajectories = len(self._demo_keys)
        self._action_dim = self._actions[0].shape[-1]
        self._state_dim = self._states[0].shape[-1]

        print(f"Loaded {self.num_trajectories} KinDER demos from {self.hdf5_path.name}")
        print(f"  State dim: {self._state_dim} (keys: {state_keys})")
        print(f"  Action dim: {self._action_dim}")
        print(f"  Image key: '{image_key}'")
        mean_len = sum(self._seq_lengths) / max(len(self._seq_lengths), 1)
        print(f"  Mean episode length: {mean_len:.1f} steps")

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def load_trajectory(self, index: int) -> TrajectoryData:
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        return TrajectoryData(
            states=self._states[index],
            actions=self._actions[index],
            seq_length=self._seq_lengths[index],
            meta={"demo_key": self._demo_keys[index]},
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1,
    ) -> torch.Tensor:
        """
        Load visual frames from the HDF5 file on demand.

        Returns:
            Tensor of shape (T, C, H, W), float32, values in [0, 1].
            WorldModelDataset resizes to the configured image_size.
        """
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        demo_key = self._demo_keys[index]

        with h5py.File(self.hdf5_path, "r") as f:
            img_ds = f["data"][demo_key]["obs"][self.image_key]
            T_ep = img_ds.shape[0]

            frame_indices = list(range(start, end, step))
            for fi in frame_indices:
                if fi >= T_ep:
                    raise ValueError(
                        f"Frame index {fi} out of range for demo '{demo_key}' "
                        f"(length={T_ep})"
                    )

            # Slice the HDF5 dataset: (T, H, W, C) uint8
            frames_np = img_ds[frame_indices]

        # (T, H, W, C) uint8  →  (T, C, H, W) float32 in [0, 1]
        frames = torch.from_numpy(frames_np.astype(np.float32) / 255.0)
        frames = rearrange(frames, "t h w c -> t c h w")
        return frames

    def get_num_trajectories(self) -> int:
        return self.num_trajectories

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim
