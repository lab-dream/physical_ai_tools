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
#
# Author: Dongyun Kim

import json
import os
from pathlib import Path
import time

from lerobot.policies.pretrained import PreTrainedPolicy
import numpy as np
from physical_ai_server.inference.act_lipo import LiPoPostOptimizer
from physical_ai_server.utils.file_utils import read_json_file
import torch


class InferenceManager:

    def __init__(
            self,
            device: str = 'cuda'):

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.policy_type = None
        self.policy_path = None
        self.policy = None
        self.lipo_params = {'enabled': False}
        self.lipo_fps = None
        self.logger = None
        self._lipo_non_act_warned = False
        self.lipo_debug_save_dir = None
        self.lipo_debug_save_every_n = 1
        self.lipo_debug_chunk_count = 0
        self.lipo_debug_raw_chunks = []
        self.lipo_debug_optimized_chunks = []

    def validate_policy(self, policy_path: str) -> bool:
        result_message = ''
        if not os.path.exists(policy_path) or not os.path.isdir(policy_path):
            result_message = f'Policy path {policy_path} does not exist or is not a directory.'
            return False, result_message

        config_path = os.path.join(policy_path, 'config.json')
        if not os.path.exists(config_path):
            result_message = f'config.json file does not exist in {policy_path}.'
            return False, result_message

        config = read_json_file(config_path)
        if (config is None or
                ('type' not in config and 'model_type' not in config)):
            result_message = f'config.json malformed or missing fields in {policy_path}.'
            return False, result_message

        available_policies = self.__class__.get_available_policies()
        policy_type = config.get('type') or config.get('model_type')
        if policy_type not in available_policies:
            result_message = f'Policy type {policy_type} is not supported.'
            return False, result_message

        self.policy_path = policy_path
        self.policy_type = policy_type
        return True, f'Policy {policy_type} is valid.'

    def load_policy(self):
        try:
            policy_cls = self._get_policy_class(self.policy_type)
            self.policy = policy_cls.from_pretrained(self.policy_path)
            self._attach_lipo_to_policy()
            return True
        except Exception as e:
            print(f'Failed to load policy from {self.policy_path}: {e}')
            return False

    def clear_policy(self):
        if hasattr(self, 'policy'):
            del self.policy
            self.policy = None
        else:
            print('No policy to clear.')

    def get_policy_config(self):
        return self.policy.config

    def configure_lipo(
            self,
            lipo_params: dict | None = None,
            fps: float | None = None,
            logger=None):
        self.lipo_params = lipo_params or {'enabled': False}
        self.lipo_fps = fps
        self.logger = logger
        self._configure_lipo_debug_saving()
        self._attach_lipo_to_policy()

    def _configure_lipo_debug_saving(self):
        self.lipo_debug_chunk_count = 0
        self.lipo_debug_raw_chunks = []
        self.lipo_debug_optimized_chunks = []
        if not self.lipo_params.get('save_debug_actions', False):
            self.lipo_debug_save_dir = None
            return

        save_dir = Path(self.lipo_params.get('debug_save_dir') or '/workspace/lipo_debug_actions')
        save_dir.mkdir(parents=True, exist_ok=True)
        self.lipo_debug_save_dir = save_dir
        self.lipo_debug_save_every_n = max(
            1,
            int(self.lipo_params.get('debug_save_every_n', 1)))
        self._log(
            'info',
            f'LiPo debug action npy saving: enabled, save_dir={save_dir}')

    def _attach_lipo_to_policy(self):
        if self.policy is None:
            return

        if not self.lipo_params.get('enabled', False):
            if hasattr(self.policy, 'act_lipo_post_optimizer'):
                delattr(self.policy, 'act_lipo_post_optimizer')
            return

        if self.policy_type != 'act':
            if not self._lipo_non_act_warned:
                self._log(
                    'warning',
                    'LiPo post-optimization: enabled but skipped because policy is not ACT')
                self._lipo_non_act_warned = True
            return

        dt = None
        if self.lipo_fps is not None and self.lipo_fps > 0:
            dt = 1.0 / float(self.lipo_fps)

        self.policy.act_lipo_post_optimizer = LiPoPostOptimizer(
            enabled=True,
            solver=self.lipo_params.get('solver', 'osqp'),
            blending_horizon=self.lipo_params.get('blending_horizon', 10),
            len_time_delay=self.lipo_params.get('len_time_delay', 0),
            dt=dt,
            epsilon_blending=self.lipo_params.get('epsilon_blending', 0.02),
            epsilon_path=self.lipo_params.get('epsilon_path', 0.003),
            osqp_eps_abs=self.lipo_params.get('osqp_eps_abs', 1e-4),
            osqp_eps_rel=self.lipo_params.get('osqp_eps_rel', 1e-4),
            osqp_max_iter=self.lipo_params.get('osqp_max_iter', 8000),
            logger=self.logger,
        )
        self._log('info', 'LiPo post-optimization: enabled for ACT')

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

    def predict(
            self,
            images: dict[str, np.ndarray],
            state: list[float],
            task_instruction: str = None) -> list:

        observation = self._preprocess(images, state, task_instruction)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
            action = action.squeeze(0).detach().cpu().numpy()

        return action

    def predict_action_chunk(
            self,
            images: dict[str, np.ndarray],
            state: list[float],
            task_instruction: str = None) -> np.ndarray:

        if self.policy_type != 'act' or not hasattr(self.policy, 'predict_action_chunk'):
            single_action = self.predict(images, state, task_instruction)
            return np.expand_dims(single_action, axis=0)

        observation = self._preprocess(images, state, task_instruction)
        with torch.inference_mode():
            raw_action_chunk = self.policy.predict_action_chunk(observation)
            optimized_action_chunk = self._maybe_apply_lipo_to_action_chunk(
                raw_action_chunk)
            self._save_lipo_debug_action_chunks(
                raw_action_chunk,
                optimized_action_chunk)
            action_chunk = optimized_action_chunk.squeeze(0).detach().cpu().numpy()

        return action_chunk

    def _maybe_apply_lipo_to_action_chunk(self, action_chunk):
        lipo_post_optimizer = getattr(self.policy, 'act_lipo_post_optimizer', None)
        if lipo_post_optimizer is None:
            return action_chunk

        return lipo_post_optimizer.maybe_optimize(
            action_chunk,
            policy_name_or_type=self.policy_type,
            fps=self.lipo_fps,
        )

    def _save_lipo_debug_action_chunks(self, raw_action_chunk, optimized_action_chunk):
        if self.lipo_debug_save_dir is None:
            return

        raw_np = self._action_chunk_to_numpy(raw_action_chunk)
        optimized_np = self._action_chunk_to_numpy(optimized_action_chunk)
        if raw_np is None or optimized_np is None:
            return
        if raw_np.shape != optimized_np.shape:
            self._log(
                'warning',
                'LiPo debug action saving skipped: raw and optimized shapes differ '
                f'({raw_np.shape} != {optimized_np.shape})')
            return

        self.lipo_debug_chunk_count += 1
        self.lipo_debug_raw_chunks.append(raw_np)
        self.lipo_debug_optimized_chunks.append(optimized_np)

        if self.lipo_debug_chunk_count % self.lipo_debug_save_every_n != 0:
            return

        raw_stack = np.stack(self.lipo_debug_raw_chunks, axis=0)
        optimized_stack = np.stack(self.lipo_debug_optimized_chunks, axis=0)
        raw_flat = raw_stack.reshape((-1, raw_stack.shape[-1]))
        optimized_flat = optimized_stack.reshape((-1, optimized_stack.shape[-1]))

        save_dir = self.lipo_debug_save_dir
        np.save(save_dir / 'raw_action_chunks.npy', raw_stack)
        np.save(save_dir / 'optimized_action_chunks.npy', optimized_stack)
        np.save(save_dir / 'log_inference_actions_raw.npy', raw_flat)
        np.save(save_dir / 'log_inference_actions_optimized.npy', optimized_flat)
        np.save(save_dir / 'log_inference_actions.npy', raw_flat)
        metadata = {
            'created_unix_s': time.time(),
            'num_chunks': int(raw_stack.shape[0]),
            'chunk_size': int(raw_stack.shape[1]),
            'action_dim': int(raw_stack.shape[2]),
            'fps': float(self.lipo_fps) if self.lipo_fps else None,
            'solver': self.lipo_params.get('solver', 'osqp'),
            'blending_horizon': int(self.lipo_params.get('blending_horizon', 10)),
            'len_time_delay': int(self.lipo_params.get('len_time_delay', 0)),
            'epsilon_blending': float(
                self.lipo_params.get('epsilon_blending', 0.02)),
            'epsilon_path': float(self.lipo_params.get('epsilon_path', 0.003)),
            'flat_raw_file': 'log_inference_actions_raw.npy',
            'flat_optimized_file': 'log_inference_actions_optimized.npy',
            'notebook_compatible_raw_file': 'log_inference_actions.npy',
        }
        (save_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))

        self._log(
            'info',
            'LiPo debug action npy saved: '
            f'save_dir={save_dir}, num_chunks={raw_stack.shape[0]}, '
            f'chunk_size={raw_stack.shape[1]}, action_dim={raw_stack.shape[2]}')

    def _action_chunk_to_numpy(self, action_chunk):
        if hasattr(action_chunk, 'detach'):
            action_np = action_chunk.detach().cpu().numpy()
        elif isinstance(action_chunk, np.ndarray):
            action_np = action_chunk
        else:
            self._log(
                'warning',
                f'LiPo debug action saving supports tensors/ndarrays only. Got {type(action_chunk)}.')
            return None

        if action_np.ndim == 3:
            if action_np.shape[0] != 1:
                self._log(
                    'warning',
                    'LiPo debug action saving supports batch size 1 only.')
                return None
            action_np = action_np[0]
        elif action_np.ndim != 2:
            self._log(
                'warning',
                f'LiPo debug action saving expected rank 2 or 3, got {action_np.shape}.')
            return None

        return np.asarray(action_np, dtype=np.float32).copy()

    def _preprocess(
            self,
            images: dict[str, np.ndarray],
            state: list,
            task_instruction: str = None) -> dict:

        observation = self._convert_images2tensors(images)
        observation['observation.state'] = self._convert_np2tensors(state)
        for key in observation.keys():
            observation[key] = observation[key].to(self.device)

        if task_instruction is not None:
            observation['task'] = [task_instruction]

        return observation

    def _convert_images2tensors(
            self,
            images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:

        processed_images = {}
        for key, value in images.items():
            image = torch.from_numpy(value)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)
            image = image.to(self.device, non_blocking=True)
            image = image.unsqueeze(0)
            processed_images['observation.images.' + key] = image

        return processed_images

    def _convert_np2tensors(
            self,
            data):
        if isinstance(data, list):
            data = np.array(data)
        tensor_data = torch.from_numpy(data)
        tensor_data = tensor_data.to(torch.float32)
        tensor_data = tensor_data.to(self.device, non_blocking=True)
        tensor_data = tensor_data.unsqueeze(0)

        return tensor_data

    def _get_policy_class(self, name: str) -> PreTrainedPolicy:
        if name == 'tdmpc':
            from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

            return TDMPCPolicy
        elif name == 'diffusion':
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            return DiffusionPolicy
        elif name == 'act':
            from lerobot.policies.act.modeling_act import ACTPolicy

            return ACTPolicy
        elif name == 'vqbet':
            from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

            return VQBeTPolicy
        elif name == 'pi0':
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            return PI0Policy
        elif name == 'pi0fast':
            from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
            return PI0FASTPolicy
        elif name == 'smolvla':
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            return SmolVLAPolicy
        # TODO: Uncomment when GrootN1Policy is implemented
        # elif name == 'groot-n1':
        #     from Isaac.groot_n1.policies.groot_n1 import GrootN1Policy
        #     return GrootN1Policy
        else:
            raise NotImplementedError(
                f'Policy with name {name} is not implemented.')

    @staticmethod
    def get_available_policies() -> list[str]:
        return [
            'tdmpc',
            'diffusion',
            'act',
            'vqbet',
            'pi0',
            'pi0fast',
            'smolvla',
        ]

    @staticmethod
    def get_saved_policies():
        import os
        import json

        home_dir = os.path.expanduser('~')
        hub_dir = os.path.join(home_dir, '.cache/huggingface/hub')
        models_folder_list = [d for d in os.listdir(hub_dir) if d.startswith('models--')]

        saved_policy_path = []
        saved_policy_type = []

        for model_folder in models_folder_list:
            model_path = os.path.join(hub_dir, model_folder)
            snapshots_path = os.path.join(model_path, 'snapshots')

            # Check if snapshots directory exists
            if os.path.exists(snapshots_path) and os.path.isdir(snapshots_path):
                # Get list of folders inside snapshots directory
                snapshot_folders = [
                    d for d in os.listdir(snapshots_path)
                    if os.path.isdir(os.path.join(snapshots_path, d))
                ]

            # Check if pretrained_model folder exists in each snapshot folder
            for snapshot_folder in snapshot_folders:
                snapshot_path = os.path.join(snapshots_path, snapshot_folder)
                pretrained_model_path = os.path.join(snapshot_path, 'pretrained_model')

                # If pretrained_model folder exists, add to saved_policies
                if os.path.exists(pretrained_model_path) and os.path.isdir(pretrained_model_path):
                    config_path = os.path.join(pretrained_model_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                if 'type' in config:
                                    saved_policy_path.append(pretrained_model_path)
                                    saved_policy_type.append(config['type'])
                                elif 'model_type' in config:
                                    saved_policy_path.append(pretrained_model_path)
                                    saved_policy_type.append(config['model_type'])
                        except (json.JSONDecodeError, IOError):
                            # If config.json cannot be read, store path only
                            print('File IO Errors : ', IOError)

        return saved_policy_path, saved_policy_type
