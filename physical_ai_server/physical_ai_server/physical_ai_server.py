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
# Author: Dongyun Kim, Seongwoo Kim

import glob
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Optional

from ament_index_python.packages import get_package_share_directory
from physical_ai_interfaces.msg import (
    HFOperationStatus,
    TaskStatus,
    TrainingStatus,
)
from physical_ai_interfaces.srv import (
    ControlHfServer,
    GetDatasetList,
    GetHFUser,
    GetModelWeightList,
    GetPolicyList,
    GetRobotTypeList,
    GetSavedPolicyList,
    GetTrainingInfo,
    GetUserList,
    SendCommand,
    SendTrainingCommand,
    SetHFUser,
    SetRobotType,
)

from physical_ai_server.communication.communicator import Communicator
from physical_ai_server.data_processing.data_manager import DataManager
from physical_ai_server.data_processing.hf_api_worker import HfApiWorker
from physical_ai_server.inference.inference_manager import InferenceManager
from physical_ai_server.inference.spline import (
    finite_difference_derivative,
    quintic_spline_multi,
)
from physical_ai_server.timer.timer_manager import TimerManager
from physical_ai_server.training.training_manager import TrainingManager
from physical_ai_server.utils.parameter_utils import (
    declare_parameters,
    load_parameters,
    log_parameters,
)

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class PhysicalAIServer(Node):
    # Define operation modes (constants taken from Communicator)

    DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'
    DEFAULT_TOPIC_TIMEOUT = 5.0  # seconds
    PUB_QOS_SIZE = 10
    TRAINING_STATUS_TIMER_FREQUENCY = 0.5  # seconds
    LIPO_PARAMETER_DEFAULTS = {
        'use_lipo': True,
        'lipo_solver': 'osqp',
        'lipo_blending_horizon': 10,
        'lipo_len_time_delay': 5,
        'lipo_epsilon_blending': 0.02,
        'lipo_epsilon_path': 0.003,
        'lipo_osqp_eps_abs': 1e-4,
        'lipo_osqp_eps_rel': 1e-4,
        'lipo_osqp_max_iter': 8000,
        'save_lipo_debug_actions': False,
        'lipo_debug_save_dir': '/workspace/lipo_debug_actions',
        'lipo_debug_save_every_n': 1,
    }
    ACTION_PUBLISH_PARAMETER_DEFAULTS = {
        'action_publish_hz': 100.0,
        'use_action_interpolation': True,
        'interpolation_method': 'quintic_spline_multi',
    }

    class RosbagNotReadyException(Exception):
        """Exception raised when rosbag recording cannot start yet."""

        pass

    def __init__(self):
        super().__init__('physical_ai_server')
        self.get_logger().info('Start Physical AI Server')

        self.params = None
        self.timer_callback_group = ReentrantCallbackGroup()
        self.total_joint_order = None
        self.on_recording = False
        self.on_inference = False
        self.lipo_params = self._declare_lipo_parameters()
        self.action_publish_params = self._declare_action_publish_parameters()
        self.action_publish_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._reset_action_publish_state()
        self._log_lipo_configuration()

        self.hf_cancel_on_progress = False

        self.robot_type_list = self.get_robot_type_list()
        self.start_recording_time: float = 0.0

        self.training_thread = None
        self.is_training = False
        self.training_status_timer = None

        self._init_core_components()

        self._init_ros_publisher()
        self._init_ros_service()

        self._setup_timer_callbacks()

        self.previous_data_manager_status = None

        self.goal_repo_id = None

    def _init_core_components(self):
        self.communicator: Optional[Communicator] = None
        self.data_manager: Optional[DataManager] = None
        self.timer_manager: Optional[TimerManager] = None
        self.heartbeat_timer: Optional[TimerManager] = None
        self.training_timer: Optional[TimerManager] = None
        self.inference_manager: Optional[InferenceManager] = None
        self.training_manager: Optional[TrainingManager] = None

        # Initialize HF API Worker
        self.hf_api_worker: Optional[HfApiWorker] = None
        self.hf_status_timer: Optional[TimerManager] = None
        self._init_hf_api_worker()

    def _init_ros_publisher(self):
        self.get_logger().info('Initializing ROS publishers...')
        pub_qos_size = 100
        self.training_status_publisher = self.create_publisher(
            TrainingStatus,
            '/training/status',
            pub_qos_size
        )

    def _init_ros_service(self):
        self.get_logger().info('Initializing ROS services...')
        service_definitions = [
            ('/task/command', SendCommand, self.user_interaction_callback),
            ('/get_robot_types', GetRobotTypeList, self.get_robot_types_callback),
            ('/set_robot_type', SetRobotType, self.set_robot_type_callback),
            ('/register_hf_user', SetHFUser, self.set_hf_user_callback),
            ('/get_registered_hf_user', GetHFUser, self.get_hf_user_callback),
            ('/get_policy_list', GetPolicyList, self.get_policy_list_callback),
            ('/get_saved_policies', GetSavedPolicyList, self.get_saved_policies_callback),
            ('/training/command', SendTrainingCommand, self.user_training_interaction_callback),
            ('/training/get_available_policy', GetPolicyList, self.get_available_list_callback),
            ('/training/get_user_list', GetUserList, self.get_user_list_callback),
            ('/training/get_dataset_list', GetDatasetList, self.get_dataset_list_callback),
            (
                '/training/get_model_weight_list',
                GetModelWeightList,
                self.get_model_weight_list_callback
            ),
            ('/huggingface/control', ControlHfServer, self.control_hf_server_callback),
            ('/training/get_training_info', GetTrainingInfo, self.get_training_info_callback),
        ]

        for service_name, service_type, callback in service_definitions:
            self.create_service(service_type, service_name, callback)

        self.get_logger().info('ROS services initialized successfully')

    def _setup_timer_callbacks(self):
        self.timer_callback_dict = {
            'collection': self._data_collection_timer_callback,
            'inference': self._inference_timer_callback
        }

    def _declare_lipo_parameters(self):
        for name, default_value in self.LIPO_PARAMETER_DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)
        return self._get_lipo_parameters()

    def _get_lipo_parameters(self):
        params = {
            name: self.get_parameter(name).value
            for name in self.LIPO_PARAMETER_DEFAULTS.keys()
        }
        return {
            'enabled': bool(params['use_lipo']),
            'solver': str(params['lipo_solver']),
            'blending_horizon': int(params['lipo_blending_horizon']),
            'len_time_delay': int(params['lipo_len_time_delay']),
            'epsilon_blending': float(params['lipo_epsilon_blending']),
            'epsilon_path': float(params['lipo_epsilon_path']),
            'osqp_eps_abs': float(params['lipo_osqp_eps_abs']),
            'osqp_eps_rel': float(params['lipo_osqp_eps_rel']),
            'osqp_max_iter': int(params['lipo_osqp_max_iter']),
            'save_debug_actions': bool(params['save_lipo_debug_actions']),
            'debug_save_dir': str(params['lipo_debug_save_dir']),
            'debug_save_every_n': int(params['lipo_debug_save_every_n']),
        }

    def _declare_action_publish_parameters(self):
        for name, default_value in self.ACTION_PUBLISH_PARAMETER_DEFAULTS.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)
        return self._get_action_publish_parameters()

    def _get_action_publish_parameters(self):
        params = {
            name: self.get_parameter(name).value
            for name in self.ACTION_PUBLISH_PARAMETER_DEFAULTS.keys()
        }
        action_publish_hz = float(params['action_publish_hz'])
        if action_publish_hz <= 0:
            self.get_logger().warning(
                'action_publish_hz must be positive; falling back to 100.0 Hz')
            action_publish_hz = 100.0
        return {
            'action_publish_hz': action_publish_hz,
            'use_action_interpolation': bool(params['use_action_interpolation']),
            'interpolation_method': str(params['interpolation_method']),
            'use_early_inference': self._is_lipo_early_inference_enabled(),
        }

    def _is_lipo_early_inference_enabled(self):
        if not self.lipo_params.get('enabled', False):
            return False

        inference_manager = getattr(self, 'inference_manager', None)
        policy_type = getattr(inference_manager, 'policy_type', None)
        if policy_type is not None and policy_type != 'act':
            return False

        return True

    def _log_lipo_configuration(self):
        if self.lipo_params.get('enabled', False):
            self.get_logger().info('LiPo post-optimization: enabled for ACT')
        else:
            self.get_logger().info('LiPo post-optimization: disabled')

    def _reset_action_publish_state(self):
        self.policy_inference_hz = 0.0
        self.action_publish_hz = self.action_publish_params['action_publish_hz']
        self.use_action_interpolation = self.action_publish_params[
            'use_action_interpolation']
        self.interpolation_method = self.action_publish_params['interpolation_method']
        self.use_early_inference = self.action_publish_params['use_early_inference']
        self.latest_action_chunk = None
        self.latest_action_chunk_id = 0
        self.current_action_pva = None
        self.next_action_pva = None
        self.action_step_counter = 0
        self.action_t_in_step = 0.0
        self.action_x0 = None
        self.action_x1 = None
        self.last_action_publish_time = None
        self.action_chunk_request_pending = False
        self.action_chunk_request_reason = ''
        self.action_chunk_request_count = 0
        self.last_published_action = None
        self._logged_action_publish_config = False
        self._logged_action_chunk_shape = False
        self._logged_early_inference_config = False

    def init_ros_params(self, robot_type):
        self.get_logger().info(f'Initializing ROS parameters for robot type: {robot_type}')
        param_names = [
            'camera_topic_list',
            'joint_topic_list',
            'observation_list',
            'joint_list',
            'rosbag_extra_topic_list',
        ]

        # Declare parameters
        declare_parameters(
            node=self,
            robot_type=robot_type,
            param_names=param_names,
            default_value=['']
        )

        # Load parameters
        self.params = load_parameters(
            node=self,
            robot_type=robot_type,
            param_names=param_names
        )

        self.joint_order_list = [
            f'joint_order.{joint_name}' for joint_name in self.params['joint_list']
        ]

        declare_parameters(
            node=self,
            robot_type=robot_type,
            param_names=self.joint_order_list,
            default_value=['']
        )

        self.joint_order = load_parameters(
            node=self,
            robot_type=robot_type,
            param_names=self.joint_order_list
        )

        self.total_joint_order = []
        for joint_list in self.joint_order.values():
            self.total_joint_order.extend(joint_list)

        # Log loaded parameters
        log_parameters(self, self.params)
        log_parameters(self, self.joint_order)

        # Initialize observation manager
        self.communicator = Communicator(
            node=self,
            operation_mode=self.operation_mode,
            params=self.params
        )

        if self.heartbeat_timer is None:
            self.heartbeat_timer = TimerManager(
                node=self,
                callback_group=self.timer_callback_group)
            self.heartbeat_timer.set_timer(
                timer_name='heartbeat',
                timer_frequency=1.0,
                callback_function=self.communicator.heartbeat_timer_callback
            )
            self.heartbeat_timer.start(timer_name='heartbeat')

        self.inference_manager = InferenceManager()
        self.get_logger().info(
            f'ROS parameters initialized successfully for robot type: {robot_type}')

    def get_training_status(self):
        msg = TrainingStatus()
        if self.training_manager is None:
            return
        try:
            current_status = self.training_manager.get_current_training_status()
            training_info = current_status.training_info
            current_step = current_status.current_step
            current_loss = current_status.current_loss
            msg.training_info = training_info
            msg.current_step = current_step
            msg.current_loss = current_loss
            msg.is_training = self.is_training
            msg.error = ''
        except Exception as e:
            msg.current_step = 0
            msg.current_loss = float('nan')
            msg.error = str(e)
            self.get_logger().error(f'Error publishing training status: {msg.error}')
            return msg
        return msg

    def init_robot_control_parameters_from_user_task(
            self,
            task_info):
        self.get_logger().info(
            'Initializing robot control parameters from user task...')
        self.data_manager = DataManager(
            save_root_path=self.DEFAULT_SAVE_ROOT_PATH,
            robot_type=self.robot_type,
            task_info=task_info
        )
        self.communicator.clear_latest_data()

        timer_frequency = float(task_info.fps)
        if timer_frequency <= 0:
            timer_frequency = 30.0
            self.get_logger().warning(
                'task_info.fps must be positive; falling back to 30.0 Hz')

        self.timer_manager = TimerManager(
            node=self,
            callback_group=self.timer_callback_group)
        self.timer_manager.set_timer(
            timer_name=self.operation_mode,
            timer_frequency=timer_frequency,
            callback_function=self.timer_callback_dict[self.operation_mode]
        )

        if self.operation_mode == 'inference':
            self.action_publish_params = self._get_action_publish_parameters()
            self._reset_action_publish_state()
            self.policy_inference_hz = timer_frequency
            if self.use_action_interpolation and self.use_early_inference:
                self._request_action_chunk_inference(reason='initial')

        self.timer_manager.start(timer_name=self.operation_mode)

        if self.operation_mode == 'inference':
            if self.use_action_interpolation:
                self.timer_manager.set_timer(
                    timer_name='action_publish',
                    timer_frequency=self.action_publish_hz,
                    callback_function=self._action_publish_timer_callback
                )
                self.timer_manager.start(timer_name='action_publish')
            self._log_action_publish_configuration()

        self.get_logger().info(
            'Robot control parameters initialized successfully')

    def _log_action_publish_configuration(self):
        publish_topics = [
            topic
            for name, topic in self.communicator.joint_topics.items()
            if 'leader' in name.lower()
        ]
        if self.use_action_interpolation:
            self.get_logger().info('Action interpolation publisher: enabled')
        else:
            self.get_logger().info('Action interpolation publisher: disabled')
        self.get_logger().info(
            'policy_inference_hz='
            f'{self.policy_inference_hz}, action_publish_hz={self.action_publish_hz}, '
            f'use_lipo={self.lipo_params.get("enabled", False)}, '
            f'use_action_interpolation={self.use_action_interpolation}, '
            f'use_early_inference={self.use_early_inference} '
            '(controlled_by=use_lipo), '
            f'early_inference_horizon='
            f'{self.lipo_params.get("blending_horizon", 0)}, '
            f'interpolation_method={self.interpolation_method}, '
            f'publish topic list={publish_topics}')

    def _stop_operation_timers(self):
        if self.timer_manager is None:
            return
        for timer_name in ('inference', 'collection', 'action_publish'):
            if timer_name in self.timer_manager._timer:
                self.timer_manager.stop(timer_name)

    def clear_parameters(self):
        if self.communicator is not None:
            self.communicator.cleanup()
            self.communicator = None

        if self.timer_manager is not None:
            self.timer_manager.stop_all()
            self.timer_manager = None

        if self.heartbeat_timer is not None:
            self.heartbeat_timer.stop(timer_name='heartbeat')
            self.heartbeat_timer = None

        if self.training_timer is not None:
            self.training_timer.stop(timer_name='training_status')
            self.training_timer = None

        self.params = None
        self.total_joint_order = None
        self.joint_order = None

    def set_hf_user_callback(self, request, response):
        request_hf_token = request.token
        try:
            if DataManager.register_huggingface_token(request_hf_token):
                self.get_logger().info('Hugging Face user token registered successfully')
                response.user_id_list = DataManager.get_huggingface_user_id()
                response.success = True
                response.message = 'Hugging Face user token registered successfully'
            else:
                self.get_logger().error('Failed to register Hugging Face user token')
                response.user_id_list = []
                response.success = False
                response.message = 'Failed to register token, Please check your token'
        except Exception as e:
            self.get_logger().error(f'Error in set_hf_user_callback: {str(e)}')
            response.user_id_list = []
            response.success = False
            response.message = f'Error in set_hf_user_callback:\n{str(e)}'

        return response

    def get_hf_user_callback(self, request, response):
        try:
            user_ids = DataManager.get_huggingface_user_id()
            if user_ids is not None:
                response.user_id_list = user_ids
                self.get_logger().info(f'Hugging Face user IDs: {user_ids}')
                response.success = True
                response.message = 'Hugging Face user IDs retrieved successfully'
            else:
                self.get_logger().error('Failed to retrieve Hugging Face user ID')
                response.user_id_list = []
                response.success = False
                response.message = 'Failed to retrieve Hugging Face user ID'
        except Exception as e:
            self.get_logger().error(f'Error in get_hf_user_callback: {str(e)}')
            response.user_id_list = []
            response.success = False
            response.message = f'Failed to retrieve Hugging Face user ID:\n{str(e)}'

        return response

    def get_robot_type_list(self):
        pkg_dir = get_package_share_directory('physical_ai_server')
        config_dir = os.path.join(pkg_dir, 'config')
        config_files = glob.glob(os.path.join(config_dir, '*.yaml'))
        config_files.sort()

        robot_type_list = []
        for config_file in config_files:
            robot_type = os.path.splitext(os.path.basename(config_file))[0]
            if robot_type.endswith('_config'):
                robot_type = robot_type[:-7]
            robot_type_list.append(robot_type)

        self.get_logger().info(f'Available robot types: {robot_type_list}')
        return robot_type_list

    def handle_rosbag_recording(self):
        try:
            current = self.data_manager.get_status()
            previous = self.previous_data_manager_status

            # Early return if no status change
            if current == previous:
                return

            handlers = {
                ('*', 'warmup'): self._handle_warmup_transition,
                ('*', 'run'): self._handle_run_transition,
                ('run', 'save'): self._handle_save_transition,
                ('run', 'stop'): self._handle_stop_transition,
                ('*', 'finish'): self._handle_finish_transition,
                ('run', 'reset'): self._handle_reset_transition,
            }

            # Try exact match first, then wildcard match
            handler = handlers.get((previous, current)) or handlers.get(('*', current))

            if handler:
                handler(previous)

            self.previous_data_manager_status = current

        except PhysicalAIServer.RosbagNotReadyException as e:
            # Expected condition: rosbag not ready yet
            self.get_logger().warn(str(e))
        except Exception as e:
            error_msg = f'Error in rosbag recording: {str(e)}'
            self.get_logger().error(traceback.format_exc())
            self.get_logger().error(error_msg)

    def _handle_warmup_transition(self, previous_status: str):
        self.get_logger().info('Preparing rosbag recording, '
                               f'previous status: {previous_status}')

        rosbag_topics = self.communicator.get_all_topics()
        self.communicator.prepare_rosbag(topics=rosbag_topics)

    def _handle_run_transition(self, previous_status: str):
        self.get_logger().info('Starting rosbag recording, '
                               f'previous status: {previous_status}')

        rosbag_path = self.data_manager.get_save_rosbag_path()

        if rosbag_path is None:
            raise PhysicalAIServer.RosbagNotReadyException(
                'Episode buffer not initialized yet, '
                'rosbag recording will start shortly')

        self.communicator.start_rosbag(rosbag_uri=rosbag_path)

    def _handle_save_transition(self, previous_status: str):
        self.get_logger().info('Stopping rosbag recording(save), '
                               f'previous status: {previous_status}')
        self.communicator.stop_rosbag()

    def _handle_stop_transition(self, previous_status: str):
        self.get_logger().info('Stopping rosbag recording(stop), '
                               f'previous status: {previous_status}')
        self.communicator.stop_rosbag()

    def _handle_finish_transition(self, previous_status: str):
        self.get_logger().info('Finishing rosbag recording, '
                               f'previous status: {previous_status}')
        self.communicator.finish_rosbag()

    def _handle_reset_transition(self, previous_status: str):
        self.get_logger().info(
                'Stopping rosbag recording and delete recorded bag, '
                f'previous status: {previous_status}')
        self.communicator.stop_and_delete_rosbag()

    def _data_collection_timer_callback(self):
        error_msg = ''
        current_status = TaskStatus()
        camera_msgs, follower_msgs, leader_msgs = self.communicator.get_latest_data()
        if camera_msgs is None:
            if time.perf_counter() - self.start_recording_time > self.DEFAULT_TOPIC_TIMEOUT:
                error_msg = 'Camera data not received within timeout period'
                self.get_logger().error(error_msg)
            else:
                self.get_logger().info('Waiting for camera data...')
                return

        elif follower_msgs is None:
            if time.perf_counter() - self.start_recording_time > self.DEFAULT_TOPIC_TIMEOUT:
                error_msg = 'Follower data not received within timeout period'
                self.get_logger().error(error_msg)
            else:
                self.get_logger().info('Waiting for follower data...')
                return

        elif leader_msgs is None:
            if time.perf_counter() - self.start_recording_time > self.DEFAULT_TOPIC_TIMEOUT:
                error_msg = 'Leader data not received within timeout period'
                self.get_logger().error(error_msg)
            else:
                self.get_logger().info('Waiting for leader data...')
                return

        try:
            camera_data, follower_data, leader_data = self.data_manager.convert_msgs_to_raw_datas(
                camera_msgs,
                follower_msgs,
                self.total_joint_order,
                leader_msgs,
                self.joint_order)

        except Exception as e:
            error_msg = f'Failed to convert messages: {str(e)}, please check the robot type again!'
            self.on_recording = False
            current_status.phase = TaskStatus.READY
            current_status.error = error_msg
            self.communicator.publish_status(status=current_status)
            self.timer_manager.stop(timer_name=self.operation_mode)
            return

        if not self.data_manager.check_lerobot_dataset(
                camera_data,
                self.total_joint_order):
            error_msg = 'Invalid repository name, Please change the repository name'
            self.get_logger().info(error_msg)

        if error_msg:
            self.on_recording = False
            current_status.phase = TaskStatus.READY
            current_status.error = error_msg
            self.communicator.publish_status(status=current_status)
            self.timer_manager.stop(timer_name=self.operation_mode)
            return

        if self.communicator.joystick_state['updated']:
            self.handle_joystick_trigger(
                joystick_mode=self.communicator.joystick_state['mode'])
            self.communicator.joystick_state['updated'] = False

        record_completed = self.data_manager.record(
            images=camera_data,
            state=follower_data,
            action=leader_data)

        current_status = self.data_manager.get_current_record_status()
        self.communicator.publish_status(status=current_status)

        if self.data_manager.should_record_rosbag2():
            self.handle_rosbag_recording()

        if record_completed:
            self.get_logger().info('Recording completed')
            current_status.phase = TaskStatus.READY
            current_status.proceed_time = int(0)
            current_status.total_time = int(0)
            self.communicator.publish_status(status=current_status)
            self.on_recording = False
            self.timer_manager.stop(timer_name=self.operation_mode)
            return

    def _inference_timer_callback(self):
        if not self.on_inference:
            return
        if self.use_action_interpolation and self.use_early_inference:
            with self.action_publish_lock:
                if not self.action_chunk_request_pending:
                    return

        if not self.inference_lock.acquire(blocking=False):
            return
        try:
            self._run_inference_once()
        finally:
            self.inference_lock.release()

    def _run_inference_once(self):
        error_msg = ''
        current_status = TaskStatus()
        camera_msgs, follower_msgs, _ = self.communicator.get_latest_data()
        if (camera_msgs is None or
                len(camera_msgs) != len(self.params['camera_topic_list'])):
            self.get_logger().info('Waiting for camera data...')
            return False
        elif follower_msgs is None:
            self.get_logger().info('Waiting for follower data...')
            return False

        try:
            camera_data, follower_data, _ = self.data_manager.convert_msgs_to_raw_datas(
                camera_msgs,
                follower_msgs,
                self.total_joint_order)
        except Exception as e:
            error_msg = f'Failed to convert messages: {str(e)}, please check the robot type again!'
            self.on_inference = False
            current_status.phase = TaskStatus.READY
            current_status.error = error_msg
            self.communicator.publish_status(status=current_status)
            self.inference_manager.clear_policy()
            self._stop_operation_timers()
            return False

        if self.inference_manager.policy is None:
            if not self.inference_manager.load_policy():
                self.get_logger().error('Failed to load policy')
                return False

        try:
            if not self.on_inference:
                self.get_logger().info('Inference mode is not active')
                current_status = self.data_manager.get_current_record_status()
                current_status.phase = TaskStatus.READY
                self.communicator.publish_status(status=current_status)
                self.inference_manager.clear_policy()
                self._stop_operation_timers()
                return False

            if self.use_action_interpolation:
                action_chunk = self.inference_manager.predict_action_chunk(
                    images=camera_data,
                    state=follower_data,
                    task_instruction=self.task_instruction[0]
                )
                self._update_action_chunk(action_chunk)
            else:
                action = self.inference_manager.predict(
                    images=camera_data,
                    state=follower_data,
                    task_instruction=self.task_instruction[0]
                )
                self.get_logger().info(
                    f'Action data: {action}')

                action_pub_msgs = self.data_manager.data_converter.tensor_array2joint_msgs(
                    action,
                    self.joint_topic_types,
                    self.joint_order
                )

                self.communicator.publish_action(
                    joint_msg_datas=action_pub_msgs
                )
            current_status = self.data_manager.get_current_record_status()
            current_status.phase = TaskStatus.INFERENCING
            self.communicator.publish_status(status=current_status)
            return True

        except Exception as e:
            self.get_logger().error(f'Inference failed, please check : {str(e)}')
            error_msg = f'Inference failed, please check : {str(e)}'
            self.on_recording = False
            self.on_inference = False
            current_status = self.data_manager.get_current_record_status()
            current_status.phase = TaskStatus.READY
            current_status.error = error_msg
            self.communicator.publish_status(status=current_status)
            self.inference_manager.clear_policy()
            self._stop_operation_timers()
            return False

    def _request_action_chunk_inference(self, reason: str) -> bool:
        with self.action_publish_lock:
            if self.action_chunk_request_pending:
                return False
            self.action_chunk_request_pending = True
            self.action_chunk_request_reason = reason
            self.action_chunk_request_count += 1
            request_count = self.action_chunk_request_count

        self.get_logger().info(
            'Early inference request: '
            f'reason={reason}, request_count={request_count}')
        return True

    def _mark_action_chunk_request_locked(self, reason: str) -> bool:
        if self.action_chunk_request_pending:
            return False
        self.action_chunk_request_pending = True
        self.action_chunk_request_reason = reason
        self.action_chunk_request_count += 1
        return True

    def _update_action_chunk(self, action_chunk):
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = np.expand_dims(action_chunk, axis=0)
        if action_chunk.ndim != 2:
            raise ValueError(f'Expected action chunk rank 2, got shape {action_chunk.shape}')

        action_pva = self._build_action_pva(action_chunk)
        with self.action_publish_lock:
            self.latest_action_chunk = action_chunk
            self.latest_action_chunk_id += 1
            if self.current_action_pva is None or not self.use_early_inference:
                self.current_action_pva = action_pva
                self.next_action_pva = None
                self.action_step_counter = 0
                self.action_t_in_step = 0.0
                self.action_x0 = action_pva[0].copy()
                self.action_x1 = action_pva[0].copy()
                self.last_action_publish_time = None
                slot = 'current'
            else:
                self.next_action_pva = action_pva
                slot = 'next'
            self.action_chunk_request_pending = False
            self.action_chunk_request_reason = ''

        if not self._logged_action_chunk_shape:
            self.get_logger().info(
                'Action chunk received: '
                f'chunk_size={action_chunk.shape[0]}, '
                f'action_dim={action_chunk.shape[1]}, '
                f'interpolation_method={self.interpolation_method}')
            self._logged_action_chunk_shape = True

        self.get_logger().info(
            'Action chunk buffered: '
            f'slot={slot}, chunk_id={self.latest_action_chunk_id}, '
            f'chunk_size={action_chunk.shape[0]}, action_dim={action_chunk.shape[1]}')

    def _build_action_pva(self, action_chunk: np.ndarray) -> np.ndarray:
        policy_dt = self._get_policy_dt()
        velocity_chunk = finite_difference_derivative(action_chunk, 1, policy_dt, 0)
        acceleration_chunk = finite_difference_derivative(
            velocity_chunk,
            1,
            policy_dt,
            0)
        return np.stack(
            (action_chunk, velocity_chunk, acceleration_chunk),
            axis=1).astype(np.float64, copy=False)

    def _get_policy_dt(self) -> float:
        if self.policy_inference_hz > 0:
            return 1.0 / float(self.policy_inference_hz)
        return 1.0 / 30.0

    def _get_early_inference_horizon(self, chunk_size: int) -> int:
        horizon = int(self.lipo_params.get('blending_horizon', 0))
        if chunk_size <= 1:
            return 0
        return min(max(horizon, 1), chunk_size - 1)

    def _action_publish_timer_callback(self):
        if not self.on_inference or not self.use_action_interpolation:
            return
        if self.communicator is None or self.data_manager is None:
            return

        now = time.perf_counter()
        request_reason = None
        with self.action_publish_lock:
            if self.current_action_pva is None:
                if not self.action_chunk_request_pending:
                    request_reason = 'initial_wait'
                    self._mark_action_chunk_request_locked(request_reason)
                action = None
            else:
                if self.last_action_publish_time is None:
                    elapsed_s = 0.0
                else:
                    elapsed_s = max(0.0, now - self.last_action_publish_time)
                self.last_action_publish_time = now
                self.action_t_in_step += elapsed_s

                policy_dt = self._get_policy_dt()
                while self.action_t_in_step >= policy_dt:
                    self.action_t_in_step -= policy_dt
                    if self._advance_action_step_locked():
                        request_reason = 'prefetch'

                action = self._compute_spline_action_at_t_locked(
                    self.action_t_in_step)

        if request_reason:
            self.get_logger().info(
                'Early inference request: '
                f'reason={request_reason}, '
                f'step_counter={self.action_step_counter}, '
                f'chunk_id={self.latest_action_chunk_id}')

        if action is None:
            return

        action_pub_msgs = self.data_manager.data_converter.tensor_array2joint_msgs(
            action,
            self.joint_topic_types,
            self.joint_order
        )
        self.communicator.publish_action(joint_msg_datas=action_pub_msgs)
        self.last_published_action = action

    def _advance_action_step_locked(self) -> bool:
        if self.current_action_pva is None:
            return False

        self.action_step_counter += 1
        chunk_size = self.current_action_pva.shape[0]
        horizon = self._get_early_inference_horizon(chunk_size)
        prefetch_idx = max(chunk_size - horizon, 1)
        requested = False

        if (self.use_early_inference and
                self.action_step_counter == prefetch_idx):
            requested = self._mark_action_chunk_request_locked('prefetch')

        if self.action_step_counter > prefetch_idx:
            if self.next_action_pva is not None:
                self.action_step_counter -= prefetch_idx
                self.current_action_pva = self.next_action_pva
                self.next_action_pva = None
                self.get_logger().info(
                    'Early inference chunk swap: '
                    f'overlap_step={self.action_step_counter}, '
                    f'prefetch_idx={prefetch_idx}, horizon={horizon}')
            else:
                self.action_step_counter = min(
                    self.action_step_counter,
                    chunk_size - 1)

        idx = min(self.action_step_counter, self.current_action_pva.shape[0] - 1)
        next_endpoint = self.current_action_pva[idx]
        if self.action_x1 is None:
            self.action_x0 = next_endpoint.copy()
            self.action_x1 = next_endpoint.copy()
        else:
            self.action_x0 = self.action_x1.copy()
            self.action_x1 = next_endpoint.copy()

        if not self._logged_early_inference_config:
            self.get_logger().info(
                'Early inference publisher: '
                f'enabled={self.use_early_inference}, '
                f'prefetch_idx={prefetch_idx}, '
                f'blending_horizon={horizon}, '
                f'policy_dt={self._get_policy_dt()}')
            self._logged_early_inference_config = True

        return requested

    def _compute_spline_action_at_t_locked(self, t: float):
        if self.action_x0 is None or self.action_x1 is None:
            return None

        x0_pos = self.action_x0[0]
        x0_vel = self.action_x0[1]
        x0_acc = self.action_x0[2]
        x1_pos = self.action_x1[0]
        x1_vel = self.action_x1[1]
        x1_acc = self.action_x1[2]
        policy_dt = self._get_policy_dt()

        method = self.interpolation_method.lower()
        if method in ('quintic_spline_multi', 'quintic_spline', 'quintic'):
            spline_action = quintic_spline_multi(
                max(0.0, min(policy_dt, float(t))),
                0.0,
                policy_dt,
                x0_pos,
                x0_vel,
                x0_acc,
                x1_pos,
                x1_vel,
                x1_acc)
            return spline_action[0].astype(np.float32, copy=False)

        if method == 'linear':
            alpha = max(0.0, min(1.0, float(t) / policy_dt))
            return ((1.0 - alpha) * x0_pos + alpha * x1_pos).astype(
                np.float32,
                copy=False)

        return x1_pos.astype(np.float32, copy=False)

    def user_training_interaction_callback(self, request, response):
        """
        Handle training command requests (START/FINISH).

        Supports both new training and resume functionality with proper validation.
        """
        try:
            if request.command == SendTrainingCommand.Request.START:
                # Initialize training components
                self.training_manager = TrainingManager()
                self.training_timer = TimerManager(node=self)
                self._setup_training_status_timer()

                # Validate training state
                if self.training_thread and self.training_thread.is_alive():
                    response.success = False
                    response.message = 'Training is already in progress'
                    return response

                # Extract resume parameters
                resume = getattr(request, 'resume', False)
                resume_model_path = getattr(request, 'resume_model_path', '').strip()

                # Log training request details
                output_folder_name = request.training_info.output_folder_name
                weight_save_root_path = TrainingManager.get_weight_save_root_path()
                self.get_logger().info(
                    f'Training request - Output: {output_folder_name}, '
                    f'Resume: {resume}, Model path: {resume_model_path}'
                )

                # Validate training configuration
                validation_result = self._validate_training_request(
                    resume, resume_model_path, output_folder_name, weight_save_root_path
                )
                if not validation_result['success']:
                    response.success = False
                    response.message = validation_result['message']
                    self._cleanup_training_on_error()
                    return response

                # Configure and start training
                self._configure_training_manager(request, resume, resume_model_path)
                self._start_training_thread()

                response.success = True
                response.message = 'Training started successfully'

            else:
                # Handle FINISH command
                if request.command == SendTrainingCommand.Request.FINISH:
                    self._stop_training()
                    response.success = True
                    response.message = 'Training stopped successfully'
                else:
                    response.success = False
                    response.message = f'Unknown command: {request.command}'

        except Exception as e:
            self.get_logger().error(f'Error in training callback: {str(e)}')
            response.success = False
            response.message = f'Training error: {str(e)}'
            self._cleanup_training_on_error()

        return response

    def _setup_training_status_timer(self):
        """Set up timer for publishing training status updates."""
        self.training_timer.set_timer(
            timer_name='training_status',
            timer_frequency=self.TRAINING_STATUS_TIMER_FREQUENCY,
            callback_function=lambda: self.training_status_publisher.publish(
                self.get_training_status()
            )
        )
        self.training_timer.start(timer_name='training_status')

    def _validate_training_request(
            self,
            resume,
            resume_model_path,
            output_folder_name,
            weight_save_root_path
    ):
        """
        Validate training request parameters.

        Returns
        -------
        dict
            {'success': bool, 'message': str}

        """
        # Check output folder conflicts for new training
        if not resume:
            output_path = weight_save_root_path / output_folder_name
            if output_path.exists():
                return {
                    'success': False,
                    'message': f'Output folder already exists: {output_path}'
                }

        # Validate resume configuration
        if resume:
            if not resume_model_path:
                return {
                    'success': False,
                    'message': 'Resume model path is required when resume=True'
                }

            # Check if resume config file exists
            full_config_path = weight_save_root_path / resume_model_path
            if not full_config_path.exists():
                return {
                    'success': False,
                    'message': f'Resume config file not found: {full_config_path}'
                }

        return {'success': True, 'message': 'Validation passed'}

    def _configure_training_manager(self, request, resume, resume_model_path):
        """Configure training manager with request parameters."""
        self.training_manager.training_info = request.training_info
        self.training_manager.resume = resume
        self.training_manager.resume_model_path = resume_model_path

    def _start_training_thread(self):
        """Start training in a separate thread."""
        def run_training():
            try:
                self.training_manager.train()
            except Exception as e:
                self.get_logger().error(f'Training error: {str(e)}')
            finally:
                self._cleanup_training_on_completion()

        self.training_thread = threading.Thread(target=run_training, daemon=True)
        self.training_thread.start()
        self.is_training = True

    def _stop_training(self):
        """Stop training gracefully."""
        self.is_training = False
        if self.training_manager:
            self.training_manager.stop_event.set()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=self.DEFAULT_TOPIC_TIMEOUT)
        self._cleanup_training_on_completion()

    def _cleanup_training_on_completion(self):
        """Cleanup training resources on normal completion."""
        self.is_training = False
        self.get_logger().info('Training completed.')
        training_status = self.get_training_status()
        self.training_status_publisher.publish(training_status)
        if self.training_manager:
            self.training_manager.stop_event.set()
        if hasattr(self, 'training_timer'):
            self.training_timer.stop('training_status')

    def _cleanup_training_on_error(self):
        """Cleanup training resources on error."""
        self.is_training = False
        training_status = self.get_training_status()
        self.training_status_publisher.publish(training_status)
        if self.training_manager:
            self.training_manager.stop_event.set()
        if hasattr(self, 'training_timer'):
            self.training_timer.stop('training_status')

    def user_interaction_callback(self, request, response):
        try:
            if request.command == SendCommand.Request.START_RECORD:
                if self.on_recording:
                    self.get_logger().info('Restarting the recording.')
                    self.data_manager.re_record()
                    response.success = True
                    response.message = 'Restarting the recording.'
                    return response

                self.get_logger().info('Start recording')
                self.operation_mode = 'collection'
                task_info = request.task_info
                self.init_robot_control_parameters_from_user_task(
                    task_info
                )

                self.start_recording_time = time.perf_counter()
                self.on_recording = True
                response.success = True
                response.message = 'Recording started'

            elif request.command == SendCommand.Request.START_INFERENCE:
                self.joint_topic_types = self.communicator.get_publisher_msg_types()
                self.operation_mode = 'inference'
                task_info = request.task_info
                self.task_instruction = task_info.task_instruction

                valid_result, result_message = self.inference_manager.validate_policy(
                    policy_path=task_info.policy_path)

                if not valid_result:
                    response.success = False
                    response.message = result_message
                    self.get_logger().error(response.message)
                    return response

                self.lipo_params = self._get_lipo_parameters()
                self.inference_manager.configure_lipo(
                    lipo_params=self.lipo_params,
                    fps=task_info.fps,
                    logger=self.get_logger()
                )

                self.init_robot_control_parameters_from_user_task(
                    task_info
                )
                if task_info.record_inference_mode:
                    self.on_recording = True
                self.on_inference = True
                self.start_recording_time = time.perf_counter()
                response.success = True
                response.message = 'Inference started'

            else:
                if not self.on_recording and not self.on_inference:
                    response.success = False
                    response.message = 'Not currently recording'
                else:
                    if request.command == SendCommand.Request.STOP:
                        self.get_logger().info('Stopping recording')
                        self.data_manager.record_stop()
                        if self.on_inference:
                            self.on_inference = False
                            self._stop_operation_timers()
                        response.success = True
                        response.message = 'Recording stopped'

                    elif request.command == SendCommand.Request.MOVE_TO_NEXT:
                        self.get_logger().info('Moving to next episode')
                        if len(request.task_info.task_instruction) > 1:
                            self.data_manager.record_next_episode()
                        else:
                            self.data_manager.record_early_save()
                        response.success = True
                        response.message = 'Moved to next episode'

                    elif request.command == SendCommand.Request.RERECORD:
                        self.get_logger().info('Re-recording current episode')
                        self.data_manager.re_record()
                        response.success = True
                        response.message = 'Re-recording current episode'

                    elif request.command == SendCommand.Request.FINISH:
                        self.get_logger().info('Terminating all operations')
                        self.data_manager.record_finish()
                        self.on_inference = False
                        self._stop_operation_timers()
                        response.success = True
                        response.message = 'All operations terminated'

                    elif request.command == SendCommand.Request.SKIP_TASK:
                        self.get_logger().info('Skipping task')
                        self.data_manager.record_skip_task()
                        response.success = True
                        response.message = 'Task skipped successfully'

        except Exception as e:
            self.get_logger().error(f'Error in user interaction: {str(e)}')
            response.success = False
            response.message = f'Error in user interaction: {str(e)}'
            return response
        return response

    def get_robot_types_callback(self, request, response):
        if self.robot_type_list is None:
            self.get_logger().error('Robot type list is not set')
            response.robot_types = []
            response.success = False
            response.message = 'Robot type list is not set'
            return response

        self.get_logger().info(f'Available robot types: {self.robot_type_list}')
        response.robot_types = self.robot_type_list
        response.success = True
        response.message = 'Robot type list retrieved successfully'
        return response

    def get_policy_list_callback(self, request, response):
        policy_list = InferenceManager.get_available_policies()
        if not policy_list:
            self.get_logger().warning('No policies available')
            response.success = False
            response.message = 'No policies available'
        else:
            self.get_logger().info(f'Available policies: {policy_list}')
            response.success = True
            response.message = 'Policy list retrieved successfully'
        response.policy_list = policy_list
        return response

    def get_available_list_callback(self, request, response):
        response.success = True
        response.message = 'Policy and device lists retrieved successfully'
        response.policy_list, response.device_list = TrainingManager.get_available_list()
        return response

    def get_user_list_callback(self, request, response):
        try:
            if not self.DEFAULT_SAVE_ROOT_PATH.exists():
                response.user_list = []
                response.success = False
                response.message = f'Path {self.DEFAULT_SAVE_ROOT_PATH} does not exist.'
                return response

            folder_names = [
                name for name in os.listdir(self.DEFAULT_SAVE_ROOT_PATH)
                if (self.DEFAULT_SAVE_ROOT_PATH / name).is_dir()
            ]

            response.user_list = folder_names
            response.success = True
            response.message = f'Found {len(folder_names)} user(s).'

        except Exception as e:
            response.user_list = []
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_dataset_list_callback(self, request, response):
        user_id = request.user_id
        user_path = self.DEFAULT_SAVE_ROOT_PATH / user_id

        try:
            if not user_path.exists() or not user_path.is_dir():
                response.dataset_list = []
                response.success = False
                response.message = f"User ID '{user_id}' does not exist at path: {user_path}"
                return response

            dataset_names = [
                name for name in os.listdir(user_path)
                if (user_path / name).is_dir()
            ]

            response.dataset_list = dataset_names
            response.success = True
            response.message = f"Found {len(dataset_names)} dataset(s) for user '{user_id}'."

        except Exception as e:
            response.dataset_list = []
            response.success = False
            response.message = f'Error: {str(e)}'

        return response

    def get_model_weight_list_callback(self, request, response):
        save_root_path = TrainingManager.get_weight_save_root_path()
        try:
            if not save_root_path.exists():
                response.success = False
                response.message = f'Path does not exist: {save_root_path}'
                response.model_weight_list = []
                return response

            model_folders = [
                f.name for f in save_root_path.iterdir()
                if f.is_dir()
            ]

            response.success = True
            response.message = f'Found {len(model_folders)} model weights'
            response.model_weight_list = model_folders

        except Exception as e:
            response.success = False
            response.message = f'Error: {str(e)}'
            response.model_weight_list = []

        return response

    def get_saved_policies_callback(self, request, response):
        saved_policy_path, saved_policy_type = InferenceManager.get_saved_policies()
        if not saved_policy_path and not saved_policy_type:
            self.get_logger().warning('No saved policies found')
            response.saved_policy_path = []
            response.saved_policy_type = []
            response.success = False
            response.message = 'No saved policies found'
        else:
            self.get_logger().info(f'Saved policies path: {saved_policy_path}')
            response.saved_policy_path = saved_policy_path
            response.saved_policy_type = saved_policy_type
            response.success = True
            response.message = 'Saved policies retrieved successfully'
        return response

    def get_training_info_callback(self, request, response):
        """
        Retrieve training configuration from a saved model.

        Loads configuration from train_config.json and populates TrainingInfo message.
        """
        try:
            # Validate request
            if not request.train_config_path:
                response.success = False
                response.message = 'train_config_path is required'
                return response

            # Clean up path (remove leading/trailing whitespace)
            train_config_path = request.train_config_path.strip()
            weight_save_root_path = TrainingManager.get_weight_save_root_path()
            config_path = weight_save_root_path / train_config_path

            # Check if config file exists
            if not config_path.exists():
                response.success = False
                response.message = f'Model config file not found: {config_path}'
                return response

            # Load and parse configuration
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                self.get_logger().info(f'Successfully loaded config from: {config_path}')

                # Populate TrainingInfo message from config
                training_info = response.training_info

                # Dataset configuration
                dataset_config = config_data.get('dataset', {})
                training_info.dataset = dataset_config.get('repo_id', '')

                # Policy configuration
                policy_config = config_data.get('policy', {})
                training_info.policy_type = policy_config.get('type', '')
                training_info.policy_device = policy_config.get('device', 'cuda')

                # Output directory (extract folder name)
                output_dir = config_data.get('output_dir', '')
                if output_dir:
                    training_info.output_folder_name = Path(output_dir).name
                else:
                    training_info.output_folder_name = ''

                # Training parameters with defaults
                training_info.seed = config_data.get('seed', 1000)
                training_info.num_workers = config_data.get('num_workers', 4)
                training_info.batch_size = config_data.get('batch_size', 8)
                training_info.steps = config_data.get('steps', 100000)
                training_info.eval_freq = config_data.get('eval_freq', 20000)
                training_info.log_freq = config_data.get('log_freq', 200)
                training_info.save_freq = config_data.get('save_freq', 1000)

                response.success = True
                response.message = \
                    f'Training configuration loaded successfully from {train_config_path}'

            except json.JSONDecodeError as e:
                response.success = False
                response.message = f'Invalid JSON in config file: {str(e)}'
                return response
            except KeyError as e:
                response.success = False
                response.message = f'Missing required field in config: {str(e)}'
                return response

        except Exception as e:
            self.get_logger().error(f'Error in get_training_info_callback: {str(e)}')
            response.success = False
            response.message = f'Failed to retrieve training info: {str(e)}'

        return response

    def set_robot_type_callback(self, request, response):
        try:
            self.get_logger().info(f'Setting robot type to: {request.robot_type}')
            self.operation_mode = 'collection'
            self.robot_type = request.robot_type
            self.clear_parameters()
            self.init_ros_params(self.robot_type)
            response.success = True
            response.message = f'Robot type set to {self.robot_type}'
            return response

        except Exception as e:
            self.get_logger().error(f'Failed to set robot type: {str(e)}')
            response.success = False
            response.message = f'Failed to set robot type: {str(e)}'
            return response

    def _init_hf_api_worker(self):
        """Initialize HF API Worker and status monitoring timer."""
        try:
            self.hf_api_worker = HfApiWorker()
            if self.hf_api_worker.start():
                self.get_logger().info('HF API Worker started successfully')
                # Initialize idle count
                self._hf_idle_count = 0
                # Initialize status monitoring timer
                self.hf_status_timer = TimerManager(node=self)
                self.hf_status_timer.set_timer(
                    timer_name='hf_status',
                    timer_frequency=2.0,
                    callback_function=self._hf_status_timer_callback
                )
                self.hf_status_timer.start(timer_name='hf_status')
                # Create publisher for HF status
                self.hf_status_publisher = self.create_publisher(
                    HFOperationStatus,
                    '/huggingface/status',
                    self.PUB_QOS_SIZE
                )
            else:
                self.get_logger().error('Failed to start HF API Worker')
        except Exception as e:
            self.get_logger().error(f'Error initializing HF API Worker: {str(e)}')

    def _hf_status_timer_callback(self):
        """Timer callback to check HF API Worker status and publish updates."""
        if self.hf_api_worker is None:
            return
        try:
            status = self.hf_api_worker.check_task_status()
            self._publish_hf_operation_status_msg(status)

            # Log status changes (avoid spamming logs)
            last_status = self._last_hf_status.get('status', 'Unknown') \
                if hasattr(self, '_last_hf_status') else 'Unknown'
            current_status = status.get('status', 'Unknown')

            if hasattr(self, '_last_hf_status') and last_status != current_status:
                self.get_logger().info(f'HF API Status changed: {last_status} -> {current_status}')

            self._last_hf_status = status
            # Idle status count and automatic shutdown
            if status.get('status', 'Unknown') == 'Idle':
                self._hf_idle_count = getattr(self, '_hf_idle_count', 0) + 1
                if self._hf_idle_count >= 5:
                    self.get_logger().info(
                        'HF API Worker idle for 5 cycles, shutting down worker and timer.')
                    self._cleanup_hf_api_worker()
            else:
                self._hf_idle_count = 0
        except Exception as e:
            self.get_logger().error(f'Error in HF status timer callback: {str(e)}')

    def _publish_hf_operation_status_msg(self, status):
        status_msg = HFOperationStatus()
        status_msg.operation = status.get('operation', 'Unknown')
        status_msg.status = status.get('status', 'Unknown')
        status_msg.repo_id = status.get('repo_id', '')
        status_msg.local_path = status.get('local_path', '')
        status_msg.message = status.get('message', '')

        progress_progress = status.get('progress', {})

        status_msg.progress_current = progress_progress.get('current', 0)
        status_msg.progress_total = progress_progress.get('total', 0)
        status_msg.progress_percentage = progress_progress.get('percentage', 0.0)

        # self.get_logger().info(f'HF API Status: {status_msg}')
        self.hf_status_publisher.publish(status_msg)

    def control_hf_server_callback(self, request, response):
        try:
            mode = request.mode
            repo_id = request.repo_id
            local_dir = request.local_dir
            repo_type = request.repo_type
            author = request.author

            if self.hf_cancel_on_progress:
                response.success = False
                response.message = 'HF API Worker is currently canceling'
                return response

            if mode == 'cancel':
                # Immediate cleanup - force stop the worker
                try:
                    self.hf_cancel_on_progress = True
                    self._cleanup_hf_api_worker_with_threading()
                    response.success = True
                    response.message = 'Cancellation started.'
                except Exception as e:
                    self.get_logger().error(f'Error during cancel: {e}')
                finally:
                    self.hf_cancel_on_progress = False
                    return response

            # Restart HF API Worker if it does not exist or is not running
            if self.hf_api_worker is None or not self.hf_api_worker.is_alive():
                self.get_logger().info('HF API Worker not running, restarting...')
                self._init_hf_api_worker()
            # Return error if the worker is busy
            if self.hf_api_worker.is_busy():
                self.get_logger().warning('HF API Worker is currently busy with another task')
                response.success = False
                response.message = 'HF API Worker is currently busy with another task'
                return response
            # Prepare request data for the worker
            request_data = {
                'mode': mode,
                'repo_id': repo_id,
                'local_dir': local_dir,
                'repo_type': repo_type,
                'author': author
            }
            # Send request to HF API Worker
            if self.hf_api_worker.send_request(request_data):
                self.get_logger().info(f'HF API request sent successfully: {mode} for {repo_id}')
                response.success = True
                response.message = f'HF API request started: {mode} for {repo_id}'
            else:
                self.get_logger().error('Failed to send request to HF API Worker')
                response.success = False
                response.message = 'Failed to send request to HF API Worker'
            return response
        except Exception as e:
            self.get_logger().error(f'Error in HF server callback: {str(e)}')
            response.success = False
            response.message = f'Error in HF server callback: {str(e)}'
            return response

    def handle_joystick_trigger(self, joystick_mode: str):
        self.get_logger().info(
            f'Joystick mode updated: {joystick_mode}')
        if self.data_manager is None:
            self.get_logger().warning(
                'Data manager is not initialized')
            return

        if not self.on_recording:
            self.get_logger().warning(
                'Not currently recording')
            return

        if joystick_mode == 'right':
            self.get_logger().info(
                'Right tact triggered - Moving to next episode')
            if len(self.data_manager.get_task_info().task_instruction) > 1:
                self.data_manager.record_next_episode()
            else:
                self.data_manager.record_early_save()
        elif joystick_mode == 'left':
            self.get_logger().info(
                'Left tact triggered - Re-record current episode')
            self.data_manager.re_record()
        elif joystick_mode == 'right_long_time':
            self.get_logger().info(
                'Right long tact triggered - Custom')
            # If you want, you can add custom functionality.
        elif joystick_mode == 'left_long_time':
            self.get_logger().info(
                'Left long tact triggered - Custom')
            # If you want, you can add custom functionality.
        else:
            self.get_logger().info(
                f'Received joystick trigger: {joystick_mode}')

    def _cleanup_hf_api_worker_with_threading(self):
        """
        Non-blocking cleanup of HF API Worker using threading.

        This method starts a separate thread to run the existing
        _cleanup_hf_api_worker method, preventing the main process.
        from blocking during shutdown.
        """
        import threading
        import time

        def cleanup_worker_thread():
            """Worker thread to run _cleanup_hf_api_worker."""
            try:
                # Call the existing cleanup method
                self._cleanup_hf_api_worker()
            except Exception as e:
                self.get_logger().error(f'Error in cleanup worker thread: {e}')

        try:
            if self.hf_status_timer is None and self.hf_api_worker is None:
                self.get_logger().info('No HF API components to cleanup')
                return

            self.get_logger().info('Starting non-blocking HF API Worker cleanup...')

            # Start cleanup thread
            cleanup_thread = threading.Thread(target=cleanup_worker_thread, daemon=True)
            cleanup_thread.start()

            # Reset references immediately (don't wait for cleanup to complete)
            self.hf_status_timer = None
            self.hf_api_worker = None

            self.get_logger().info('HF API Worker cleanup thread started')

            # Publish cancel status messages
            for i in range(3):
                self._publish_hf_operation_status_msg({
                    'status': 'Idle',
                    'operation': 'stop',
                    'repo_id': '',
                    'local_path': '',
                    'message': 'Canceled by stop command',
                    'progress': {
                        'current': 0,
                        'total': 0,
                        'percentage': 0.0,
                    }
                })
                time.sleep(0.5)

        except Exception as e:
            self.get_logger().error(
                f'Error starting non-blocking HF API Worker cleanup: {str(e)}'
            )
            # Fallback to blocking cleanup if threading fails
            self._cleanup_hf_api_worker()
        finally:
            self.hf_cancel_on_progress = False

    def _cleanup_hf_api_worker(self):
        """Cleanup HF API Worker and related timers."""
        try:
            if self.hf_status_timer is not None:
                self.hf_status_timer.stop(timer_name='hf_status')
                self.hf_status_timer = None

            if self.hf_api_worker is not None:
                self.hf_api_worker.stop()
                self.hf_api_worker = None

            self.get_logger().info('HF API Worker cleaned up successfully')
        except Exception as e:
            self.get_logger().error(f'Error cleaning up HF API Worker: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = PhysicalAIServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup HF API Worker before destroying node
        node._cleanup_hf_api_worker()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
