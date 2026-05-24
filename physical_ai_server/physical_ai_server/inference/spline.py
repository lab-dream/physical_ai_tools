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

import numpy as np


def finite_difference_derivative(
        values: np.ndarray,
        order: int,
        dt: float,
        axis: int = 0) -> np.ndarray:
    if order < 1:
        return values
    if values.shape[axis] < 2 or dt <= 0:
        return np.zeros_like(values)

    derivative = np.gradient(values, dt, axis=axis)
    for _ in range(order - 1):
        if derivative.shape[axis] < 2:
            return np.zeros_like(derivative)
        derivative = np.gradient(derivative, dt, axis=axis)
    return derivative


def quintic_spline_multi(
        t: float,
        t_0: float,
        t_f: float,
        x_0: np.ndarray,
        x_dot_0: np.ndarray,
        x_ddot_0: np.ndarray,
        x_f: np.ndarray,
        x_dot_f: np.ndarray,
        x_ddot_f: np.ndarray) -> np.ndarray:
    x_0 = np.asarray(x_0, dtype=np.float64)
    x_dot_0 = np.asarray(x_dot_0, dtype=np.float64)
    x_ddot_0 = np.asarray(x_ddot_0, dtype=np.float64)
    x_f = np.asarray(x_f, dtype=np.float64)
    x_dot_f = np.asarray(x_dot_f, dtype=np.float64)
    x_ddot_f = np.asarray(x_ddot_f, dtype=np.float64)

    if t <= t_0:
        return np.stack((x_0, x_dot_0, x_ddot_0), axis=0)
    if t >= t_f:
        return np.stack((x_f, x_dot_f, x_ddot_f), axis=0)

    duration = float(t_f - t_0)
    if duration <= 0:
        return np.stack((x_f, x_dot_f, x_ddot_f), axis=0)

    elapsed = float(t - t_0)
    a0 = x_0
    a1 = x_dot_0
    a2 = 0.5 * x_ddot_0

    rhs = np.stack((
        x_f - x_0 - x_dot_0 * duration - 0.5 * x_ddot_0 * duration**2,
        x_dot_f - x_dot_0 - x_ddot_0 * duration,
        x_ddot_f - x_ddot_0,
    ), axis=0)
    system = np.array([
        [duration**3, duration**4, duration**5],
        [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
        [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
    ], dtype=np.float64)

    a3, a4, a5 = np.linalg.solve(system, rhs)
    position = (
        a0 + a1 * elapsed + a2 * elapsed**2 +
        a3 * elapsed**3 + a4 * elapsed**4 + a5 * elapsed**5
    )
    velocity = (
        a1 + 2.0 * a2 * elapsed +
        3.0 * a3 * elapsed**2 + 4.0 * a4 * elapsed**3 +
        5.0 * a5 * elapsed**4
    )
    acceleration = (
        2.0 * a2 +
        6.0 * a3 * elapsed + 12.0 * a4 * elapsed**2 +
        20.0 * a5 * elapsed**3
    )
    return np.stack((position, velocity, acceleration), axis=0)
