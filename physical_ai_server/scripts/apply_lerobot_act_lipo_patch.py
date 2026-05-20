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

import argparse
from pathlib import Path


TEMPORAL_ORIGINAL = """        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action
"""

TEMPORAL_PATCHED = """        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            actions = self._maybe_apply_act_lipo(actions)
            action = self.temporal_ensembler.update(actions)
            return action
"""

QUEUE_ORIGINAL = """        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
"""

QUEUE_PATCHED = """        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            actions = self._maybe_apply_act_lipo(actions)
            actions = actions[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
"""

METHOD_INSERT_BEFORE = """    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
"""

LIPO_METHOD = """    def _maybe_apply_act_lipo(self, actions: Tensor) -> Tensor:
        lipo_post_optimizer = getattr(self, "act_lipo_post_optimizer", None)
        if lipo_post_optimizer is None:
            return actions
        return lipo_post_optimizer.maybe_optimize(
            actions,
            policy_name_or_type=getattr(self, "name", self.__class__.__name__),
        )

"""


def resolve_modeling_act_file(args) -> Path:
    if args.modeling_act_file is not None:
        return args.modeling_act_file

    if args.lerobot_root is not None:
        return args.lerobot_root / 'src' / 'lerobot' / 'policies' / 'act' / 'modeling_act.py'

    import lerobot.policies.act.modeling_act as modeling_act
    return Path(modeling_act.__file__)


def replace_once(text: str, original: str, patched: str, description: str) -> tuple[str, bool]:
    if patched in text:
        return text, False
    if original not in text:
        raise RuntimeError(f'Unable to patch LeRobot ACT policy: missing {description} anchor.')
    return text.replace(original, patched, 1), True


def apply_patch(modeling_act_file: Path) -> bool:
    text = modeling_act_file.read_text()
    changed = False

    text, did_change = replace_once(
        text,
        TEMPORAL_ORIGINAL,
        TEMPORAL_PATCHED,
        'temporal ensemble action chunk',
    )
    changed = changed or did_change

    text, did_change = replace_once(
        text,
        QUEUE_ORIGINAL,
        QUEUE_PATCHED,
        'queued action chunk',
    )
    changed = changed or did_change

    if LIPO_METHOD not in text:
        if METHOD_INSERT_BEFORE not in text:
            raise RuntimeError('Unable to patch LeRobot ACT policy: missing method insertion anchor.')
        text = text.replace(METHOD_INSERT_BEFORE, LIPO_METHOD + METHOD_INSERT_BEFORE, 1)
        changed = True

    if changed:
        modeling_act_file.write_text(text)

    return changed


def main():
    parser = argparse.ArgumentParser(
        description='Apply the Physical AI ACT LiPo integration patch to an installed LeRobot tree.'
    )
    parser.add_argument(
        '--lerobot-root',
        type=Path,
        help='Path to the LeRobot repository root, for editable installs.',
    )
    parser.add_argument(
        '--modeling-act-file',
        type=Path,
        help='Path to lerobot/policies/act/modeling_act.py.',
    )
    args = parser.parse_args()

    modeling_act_file = resolve_modeling_act_file(args)
    if not modeling_act_file.exists():
        raise FileNotFoundError(f'LeRobot ACT policy file not found: {modeling_act_file}')

    changed = apply_patch(modeling_act_file)
    if changed:
        print(f'Applied ACT LiPo patch to {modeling_act_file}')
    else:
        print(f'ACT LiPo patch already applied to {modeling_act_file}')


if __name__ == '__main__':
    main()
