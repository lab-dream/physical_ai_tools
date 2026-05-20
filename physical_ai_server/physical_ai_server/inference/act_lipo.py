#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np


class LiPoPostOptimizer:
    """ACT-only LiPo post-optimizer for full action chunks."""

    DEFAULT_DT = 1.0 / 30.0

    def __init__(
            self,
            enabled: bool = False,
            solver: str = 'osqp',
            blending_horizon: int = 10,
            len_time_delay: int = 0,
            dt: float | None = None,
            epsilon_blending: float = 0.02,
            epsilon_path: float = 0.003,
            osqp_eps_abs: float = 1e-4,
            osqp_eps_rel: float = 1e-4,
            osqp_max_iter: int = 8000,
            logger: Any = None):
        self.enabled = enabled
        self.solver = solver
        self.blending_horizon = blending_horizon
        self.len_time_delay = len_time_delay
        self.dt = dt
        self.epsilon_blending = epsilon_blending
        self.epsilon_path = epsilon_path
        self.osqp_eps_abs = osqp_eps_abs
        self.osqp_eps_rel = osqp_eps_rel
        self.osqp_max_iter = osqp_max_iter
        self.logger = logger

        self.optimizer = None
        self.optimizer_key = None
        self.prev_chunk = None
        self.warned = set()

    def maybe_optimize(
            self,
            action_chunk,
            policy_name_or_type: str | None = None,
            fps: float | None = None):
        if not self.enabled:
            return action_chunk

        if not self._is_act(policy_name_or_type):
            self._warn_once(
                'non_act',
                'LiPo is enabled, but the current policy is not ACT. Skipping LiPo.')
            return action_chunk

        converted = self._to_numpy_2d(action_chunk)
        if converted is None:
            return action_chunk

        action_np, restore_fn = converted
        dt = self._resolve_dt(fps)

        try:
            self._ensure_optimizer(action_np.shape[0], action_np.shape[1], dt)
        except RuntimeError:
            raise
        except Exception as e:
            self._warn_once(
                'init_failed',
                f'LiPo initialization failed; using raw ACT action chunk. Error: {e}')
            return action_chunk

        try:
            optimized_np, solve_info = self._solve(action_np)
        except Exception as e:
            self._warn_once(
                'solve_exception',
                f'LiPo solve failed; using raw ACT action chunk. Error: {e}')
            return action_chunk

        if optimized_np is None:
            self._warn_once(
                'solve_failed',
                f'LiPo solve failed; using raw ACT action chunk. Error: {solve_info}')
            return action_chunk

        if optimized_np.shape != action_np.shape:
            self._warn_once(
                'shape_changed',
                'LiPo output shape changed unexpectedly; using raw ACT action chunk.')
            return action_chunk

        self.prev_chunk = optimized_np.copy()
        return restore_fn(optimized_np)

    def _solve(self, action_np: np.ndarray):
        if self.prev_chunk is None or len(self.prev_chunk) < 4:
            return self.optimizer.solve(action_np, action_np, 0)

        len_past_actions = min(
            self.optimizer.B,
            max(0, len(self.prev_chunk) - self.optimizer.JM)
        )
        return self.optimizer.solve(action_np, self.prev_chunk, len_past_actions)

    def _ensure_optimizer(self, chunk_size: int, action_dim: int, dt: float):
        blending_horizon = min(
            self.blending_horizon,
            max(chunk_size - 1, 1)
        )
        optimizer_key = (
            chunk_size,
            action_dim,
            dt,
            self.solver,
            blending_horizon,
            self.len_time_delay,
            self.epsilon_blending,
            self.epsilon_path,
            self.osqp_eps_abs,
            self.osqp_eps_rel,
            self.osqp_max_iter,
        )
        if self.optimizer is not None and self.optimizer_key == optimizer_key:
            return

        try:
            from action_lipo import ActionLiPo
        except ModuleNotFoundError as e:
            raise RuntimeError(
                'LiPo is enabled for ACT, but action-lipo is not installed. '
                'Install it with: pip install action-lipo osqp scipy'
            ) from e

        self.optimizer = ActionLiPo(
            solver=self.solver,
            chunk_size=chunk_size,
            blending_horizon=blending_horizon,
            action_dim=action_dim,
            len_time_delay=self.len_time_delay,
            dt=dt,
            epsilon_blending=self.epsilon_blending,
            epsilon_path=self.epsilon_path,
            osqp_eps_abs=self.osqp_eps_abs,
            osqp_eps_rel=self.osqp_eps_rel,
            osqp_max_iter=self.osqp_max_iter,
        )
        self.optimizer_key = optimizer_key
        self.prev_chunk = None
        self._log(
            'info',
            'LiPo initialized: '
            f'chunk_size={chunk_size}, action_dim={action_dim}, '
            f'blending_horizon={blending_horizon}, dt={dt}, solver={self.solver}')

    def _to_numpy_2d(self, action_chunk):
        original_shape = tuple(action_chunk.shape)
        is_torch_tensor = (
            hasattr(action_chunk, 'detach') and
            hasattr(action_chunk, 'new_tensor')
        )

        if is_torch_tensor:
            action_np = action_chunk.detach().cpu().numpy()

            def restore_fn(optimized_np):
                restored_np = self._restore_shape(optimized_np, original_shape)
                return action_chunk.new_tensor(restored_np)

        elif isinstance(action_chunk, np.ndarray):
            action_np = action_chunk
            original_dtype = action_chunk.dtype

            def restore_fn(optimized_np):
                restored_np = self._restore_shape(optimized_np, original_shape)
                return restored_np.astype(original_dtype, copy=False)

        else:
            self._warn_once(
                'unsupported_type',
                f'LiPo supports torch.Tensor and numpy.ndarray only. Got {type(action_chunk)}.')
            return None

        if action_np.ndim == 3:
            if action_np.shape[0] != 1:
                self._warn_once(
                    'batch_gt_one',
                    'LiPo supports only batch size 1; skipping LiPo.')
                return None
            action_np = action_np[0]
        elif action_np.ndim != 2:
            self._warn_once(
                'invalid_rank',
                f'LiPo requires action chunk rank 2 or [1, T, D]. Got shape {original_shape}.')
            return None

        if action_np.shape[0] < 1 or action_np.shape[1] < 1:
            self._warn_once(
                'invalid_shape',
                f'LiPo requires non-empty action chunks. Got shape {original_shape}.')
            return None

        return np.ascontiguousarray(action_np, dtype=np.float64), restore_fn

    def _restore_shape(self, optimized_np: np.ndarray, original_shape: tuple[int, ...]):
        if len(original_shape) == 3:
            return optimized_np.reshape((1,) + optimized_np.shape)
        return optimized_np.reshape(original_shape)

    def _resolve_dt(self, fps: float | None) -> float:
        if fps is not None and fps > 0:
            return 1.0 / float(fps)
        if self.dt is not None and self.dt > 0:
            return float(self.dt)
        return self.DEFAULT_DT

    def _is_act(self, policy_name_or_type: str | None) -> bool:
        if policy_name_or_type is None:
            return False
        normalized = str(policy_name_or_type).lower()
        return normalized == 'act' or normalized == 'actpolicy'

    def _warn_once(self, key: str, message: str):
        if key in self.warned:
            return
        self.warned.add(key)
        self._log('warning', message)

    def _log(self, level: str, message: str):
        if self.logger is None:
            print(message)
            return

        log_fn = getattr(self.logger, level, None)
        if log_fn is None and level == 'warning':
            log_fn = getattr(self.logger, 'warn', None)
        if log_fn is None:
            print(message)
            return

        log_fn(message)
