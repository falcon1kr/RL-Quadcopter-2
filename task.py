import numpy as np
from physics_sim import PhysicsSim

from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 19
        self.action_low = 0
        self.action_high = 2000
        self.action_size = 4

        self.init_pos = self.current_pos
        self.last_pos = self.init_pos
        self.init_distance = np.linalg.norm(target_pos - self.init_pos)
        self.last_distance = self.init_distance

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.proximity = 1.0
        self.perimeter = 5.0
        self.goal_dist = 0.25
        self.speed_limit = 0.5
        self.accel_limit = 1.0
        self.angular_speed_limit = 0.1
        self.near_stop_speed = 0.1

    @property
    def current_pos(self):
        return self.sim.pose[:3]

    @property
    def current_state(self):
        current_distance = self.current_distance
        # return np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v])
        return np.concatenate([
            self.sim.pose[3:],
            self.sim.v,
            self.sim.linear_accel,
            self.sim.angular_v,
            self.sim.angular_accels,
            [current_distance],
            ((self.target_pos - self.current_pos) / current_distance) if self.goal_dist < current_distance else [0, 0, 0]
        ])
        # return np.concatenate([self.sim.pose[:3]])

    @property
    def speed(self):
        return np.linalg.norm(self.sim.v)

    @property
    def current_distance(self):
        return np.linalg.norm(self.target_pos - self.current_pos)

    @property
    def current_distance_square(self):
        return (self.target_pos - self.current_pos).sum()

    @property
    def future_pos(self):
        return self.current_pos[:3] + self.sim.v * self.sim.dt + 0.5 * self.sim.linear_accel * self.sim.dt * self.sim.dt

    @property
    def future_rot(self):
        angles = self.sim.pose[3:] + self.sim.angular_v * self.sim.dt + 0.5 * self.sim.angular_accels * self.sim.dt * self.sim.dt
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        return angles

    def clip(self, min_val, val, max_val = None):
        max_val = -min_val if max_val is None else max_val
        return min([max_val, max([min_val, val])])

    def gaussian(self, mean, std_dev, x):
        return np.exp(-((x - mean)**2) / (2 * std_dev * std_dev))

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        reward = 0
        current_distance = self.current_distance
        speed = self.speed
        accel = self.sim.linear_accel
        accel_amount = np.linalg.norm(accel)
        angular_speed = np.linalg.norm(self.sim.angular_v)

        if self.sim.done:
            if self.sim.time < self.sim.runtime:
                return -1

        reward += 0.005 * -1 * (1 / (1 + np.exp(-2.5 * (speed - self.speed_limit) + 5)) - 1)
        reward += 0.001 * -1 * (1 / (1 + np.exp(-2.5 * (accel_amount - self.accel_limit) + 5)) - 1)
        reward += 0.001 * -1 * (1 / (1 + np.exp(-2.5 * (angular_speed - self.angular_speed_limit) + 5)) - 1)

        if 7 * np.pi / 4 < self.sim.pose[3] or self.sim.pose[3] < np.pi / 4:
            reward += 0.001

        if 7 * np.pi / 4 < self.sim.pose[4] or self.sim.pose[4] < np.pi / 4:
            reward += 0.001

        for i in range(3):
            curr_axis_dist = abs(self.target_pos[i] - self.sim.pose[i])
            init_axis_dist = abs(self.target_pos[i] - self.init_pos[i])

            # reward += 0.03 * -1 * (1 / (1 + np.exp(-2.5 * (curr_axis_dist / init_axis_dist) + 5)) - 1) * (3 if i == 2 else 1)
            # reward += 0.03 * -1 * (1 / (1 + np.exp(-2.5 * (curr_axis_dist / 100.0) + 5)) - 1) * (3 if i == 2 else 1)
            reward += 0.03 * -1 * (1 / (1 + np.exp(-5 * (curr_axis_dist / self.perimeter) + 0.5)) - 1) * (3 if i == 2 else 1)

            if ((self.current_pos[i] < (self.target_pos[i] - self.goal_dist)) and (0 <= self.sim.v[i])) \
                    or (((self.target_pos[i] + self.goal_dist) < self.current_pos[i]) and (self.sim.v[i] <= 0)):
                reward += 0.02 * (3 if i == 2 and 0 < self.sim.v[i] else 1)

            if (curr_axis_dist <= self.goal_dist) and (abs(self.sim.v[i]) <= self.near_stop_speed):
                reward += 0.05

        # ============

        # n, v = self.init_pos - self.target_pos, curr_pos - self.target_pos
        # sway = np.linalg.norm(curr_pos - (self.target_pos + np.dot(v, n) / np.dot(n, n) * n))

        # if self.current_distance == 0:
        #     return 100

        # if self.init_distance == 0:
        #     sway = 0
        # else:
        #     u = self.target_pos - self.init_pos
        #     v = self.current_pos - self.init_pos

        #     sway = np.dot(np.cross(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), v)

        # reward -= sway
        # reward += np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.speed * self.current_distance + 0.0001) # reward on velocity's directional correctness + speed

        # print('%s %s %s' % (self.target_pos, self.current_pos, self.current_distance))
        # penalty = 0
        # penalty += min([5., self.current_distance / (self.init_distance + .01) * 2.5]) # penalize the distance
        # penalty += 1 if self.speed > 10 else 0
        # penalty += np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.current_distance * self.speed) if (self.current_distance > 0 and self.speed > 0) else 0 # reward on velocity's directional correctness + speed
        # penalty = penalty / 7.
        # # reward += max(-2.5, min(2.5, ((self.last_distance - self.current_distance) / (self.init_distance + .01)) * 2.5)) # reward if getting closer to target, penalize if getting furter from target

        # reward = max([-1., -penalty])

        # reward = 0.5 - max([-5, self.current_distance]) / 5.
        # if self.sim.done:
        #     reward -= self.sim.runtime - self.sim.time


        # if (self.current_distance > 0 and self.speed > 0):
        #     reward += max([-2, min([2, np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.current_distance)])])

        # reward += -min([5, (self.current_distance * self.current_distance)])

        # if self.current_distance < 2:
        #     reward += 3. - self.current_distance

        # reward = 0.4 + (reward / 10) * 0.6

        # if self.sim.done:
        #     reward = min([0, (self.sim.time - self.sim.runtime) * 5]) + max([-self.sim.runtime * 2., - self.current_distance])

        # reward = 0.3 + 0.5 * self.sim.time / self.sim.runtime

        # ============

        # curr_dist = self.current_distance

        # if self.sim.done:
        #     time_left = self.sim.runtime - self.sim.time
        #     reward = 0
        #     reward += -max([0, (np.log10(max([0, time_left]) / self.sim.runtime) + 1)]) * 50 if time_left > 0 else 25
        #     reward += -min([50, (max([0, curr_dist - self.init_distance]) / self.init_distance)**2]) if self.current_pos[2] > 0 else -50
        #     return reward

        # reward = 0.5

        # future_pos = self.future_pos

        # if curr_dist < self.init_distance:
        #     reward += 0.25 + 0.25 * np.log((self.init_distance - curr_dist) / self.init_distance + 1)

        # future_distance = np.linalg.norm(self.target_pos - future_pos)
        # dist_delta = abs(future_distance - curr_dist)

        # if future_distance < curr_dist:
        #     reward += 0.5
        # else:
        #     reward += -0.75

        # reward = min([1, max([-1, reward])])

        # ============

        # reward = 0.5

        # future_pos = self.future_pos

        # reward += -max([-50, min([50, (max(curr_dist - self.init_distance) / self.init_distance)**3])]) if self.current_pos[2] > 0 else -50
        # reward = min([1, max([-1, reward])])

        # if self.sim.done:
        #     reward += -min([50, (max([0, curr_dist - self.init_distance]) / self.init_distance)**2]) if self.current_pos[2] > 0 else -50

        # ============

        # reward = 0

        # if self.sim.done:
        #     time_left_threshold = 2.
        #     time_left = max([0, self.sim.runtime - self.sim.time])

        #     if time_left > time_left_threshold:
        #         reward += -max([0, (np.log10(time_left / self.sim.runtime) + 1)]) * 50 
        #     else:
        #         reward += 25 * (self.sim.runtime - time_left_threshold - time_left) / self.sim.runtime

        #     return reward

        # future_pos = self.future_pos

        # reward += 0.3

        # for i in range(3):
        #     initial_axis_diff = (self.target_pos[i] - self.init_pos[i])

        #     curr_axis_diff = abs(self.target_pos[i] - self.current_pos[i])
        #     future_axis_diff = abs(self.target_pos[i] - self.future_pos[i])

        #     curr_axis_change = initial_axis_diff - curr_axis_diff
        #     curr_axis_diff_delta_ratio = curr_axis_change / (initial_axis_diff if initial_axis_diff else 0.00001)
        #     reward += 0.01 * self.clip(-30, curr_axis_diff_delta_ratio, 20)

        #     future_axis_change = curr_axis_diff - future_axis_diff
        #     future_axis_diff_delta_ratio = future_axis_change / (initial_axis_diff if initial_axis_diff else 0.00001)
        #     reward += 0.01 * self.clip(-30, future_axis_diff_delta_ratio, 20)

        # if abs(self.sim.pose[0]) > np.pi / 4 or abs(self.sim.pose[1]) > np.pi / 4:
        #     reward += -0.2

        # ============

        # reward = 0

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])

        #     reward += (-(2.5 / (1 + np.exp(-5 * ((time_left / self.sim.runtime) - 0.2) )) - 1) + 0.5) * 5

        #     return reward

        # future_pos = self.future_pos

        # reward += 0.1

        # for i in range(2):

        #     initial_axis_dist = (self.target_pos[i] - self.init_pos[i])

        #     curr_axis_dist = abs(self.target_pos[i] - self.current_pos[i])
        #     future_axis_dist = abs(self.target_pos[i] - self.future_pos[i])

        #     reward += 0.5 * (2 / (1 + np.exp(-(initial_axis_dist - curr_axis_dist) / initial_axis_dist)) - 1)
        #     reward += 0.5 * (2 / (1 + np.exp(-(curr_axis_dist - future_axis_dist) / initial_axis_dist)) - 1)

        # ============

        # reward = 0

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])

        #     reward += (-(2.5 / (1 + np.exp(-5 * ((time_left / self.sim.runtime) - 0.2) )) - 1) + 0.5) * 15

        #     return reward

        # future_pos = self.future_pos

        # reward += 0.3

        # future_distance = np.linalg.norm(self.target_pos - self.future_pos)
        # current_distance = self.current_distance

        # reward += 0.3 * (2 / (1 + np.exp(-(self.last_distance - current_distance) / self.init_distance)) - 1)
        # reward += 0.3 * (2 / (1 + np.exp(-(current_distance - future_distance) / self.init_distance)) - 1)

        # ============

        # reward = 0
        # current_distance = self.current_distance

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])
        #     reward += -(2 / (1 + np.exp(- 5 * time_left / self.sim.runtime)) - 1)
        #     return reward

        # future_pos = self.future_pos

        # reward += 0.6

        # future_distance = np.linalg.norm(self.target_pos - self.future_pos)

        # reward += 0.3 * (2 / (1 + np.exp(-(self.init_distance - current_distance) / self.init_distance)) - 1)
        # reward += 0.3 * (2 / (1 + np.exp(-(current_distance - future_distance) / self.init_distance)) - 1)

        # speed_limit = 2.5
        # speed = self.speed
        # future_v = self.sim.v + self.sim.linear_accel * self.sim.dt
        # future_speed = np.linalg.norm(future_v)

        # if speed > speed_limit:
        #     reward += -0.1 * (2 / (1 + np.exp(-2 * ((speed - speed_limit) / speed_limit))) - 1)
        # else:
        #     reward += 0.1 * (2 / (1 + np.exp(-5 * (speed / speed_limit))) - 1)

        # if future_speed > speed_limit:
        #     reward += -0.1 * (2 / (1 + np.exp(-2 * ((future_speed - speed_limit) / speed_limit))) - 1)
        # else:
        #     reward += 0.1 * (2 / (1 + np.exp(-5 * (future_speed / speed_limit))) - 1)

        # ============

        # reward = 0
        # current_distance = self.current_distance
        # future_pos = self.future_pos
        # future_distance = np.linalg.norm(self.target_pos - future_pos)
        # proximity = 3.0

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])
        #     reward += -(2 / (1 + np.exp(- 5 * (time_left / self.sim.runtime))) - 1) / 2 + 0.5
        #     return reward

        # reward += 0.05 * (cosine_similarity([self.sim.v], [self.target_pos - self.current_pos])[0][0] / 2 + 0.5)
        # reward += 0.1 * -1 * (1 / (1 + np.exp(-5 * (current_distance / proximity) + 5)) - 1)
        # reward += 0.025 * -1 * (1 / (1 + np.exp(-5 * (current_distance - future_distance) / min([proximity, self.init_distance]))) - 1)

        # speed = self.speed
        # speed_limit = 0.5

        # if speed > speed_limit:
        #     reward += -0.01 * (2 / (1 + np.exp(-3 * ((speed - speed_limit) / speed_limit))) - 1)

        # accel = self.sim.linear_accel
        # accel_amount = np.linalg.norm(accel)
        # accel_limit = 1.0

        # if accel_amount > accel_limit:
        #     reward += -0.01 * (2 / (1 + np.exp(-3 * ((accel_amount - accel_limit) / accel_limit))) - 1)

        # angular_speed = np.linalg.norm(self.sim.angular_v)
        # angular_speed_limit = 0.5

        # if angular_speed > angular_speed_limit:
        #     reward += -0.01 * (2 / (1 + np.exp(-3 * ((angular_speed - angular_speed_limit) / angular_speed_limit))) - 1)

        # ============

        # reward = 0
        # current_distance = self.current_distance
        # future_pos = self.future_pos
        # future_distance = np.linalg.norm(self.target_pos - future_pos)
        # proximity = 3.0

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])
        #     reward += -(2 / (1 + np.exp(-5 * (time_left / self.sim.runtime))) - 1) * 10
        #     return reward

        # reward += 0.1
        # reward += 0.1 * (cosine_similarity([self.sim.v], [self.target_pos - self.current_pos])[0][0] / 2 + 0.5)
        # reward += 0.1 * -1 * (1 / (1 + np.exp(-4 * (future_distance / proximity) + 5)) - 1)

        # speed = self.speed
        # speed_limit = 0.5

        # reward += 0.01 * -1 * (1 / (1 + np.exp(-10 * (speed - speed_limit) + 5)) - 1)

        # accel = self.sim.linear_accel
        # accel_amount = np.linalg.norm(accel)
        # accel_limit = 1.0

        # reward += 0.01 * -1 * (1 / (1 + np.exp(-5 * (accel_amount - accel_limit) + 5)) - 1)

        # angular_speed = np.linalg.norm(self.sim.angular_v)
        # angular_speed_limit = 0.1

        # reward += 0.01 * -1 * (1 / (1 + np.exp(-10 * (angular_speed - angular_speed_limit) + 5)) - 1)

        # ============

        # reward = 0
        # current_distance = self.current_distance
        # future_pos = self.future_pos
        # future_distance = np.linalg.norm(self.target_pos - future_pos)
        # proximity = 2.0
        # perimeter = 5.0

        # speed = self.speed
        # speed_limit = 0.5

        # accel = self.sim.linear_accel
        # accel_amount = np.linalg.norm(accel)
        # accel_limit = 1.0

        # angular_speed = np.linalg.norm(self.sim.angular_v)
        # angular_speed_limit = 0.1

        # if self.sim.done:
        #     return min([0, (self.sim.runtime - self.sim.time)]) / self.sim.runtime * 2 - 1

        # if (cosine_similarity([self.sim.v], [self.target_pos - self.current_pos])[0][0] > 0.5):
        #     reward += 0.1

        # if (cosine_similarity([self.sim.v + self.sim.linear_accel * self.sim.dt], [future_pos - self.current_pos])[0][0] > 0.5):
        #     reward += 0.1

        # if self.current_distance <= perimeter:
        #     reward += 0.075

        # if self.current_distance <= proximity:
        #     reward += 0.5

        # if speed < speed_limit:
        #     reward += 0.01

        # if accel_amount < accel_limit:
        #     reward += 0.01

        # if angular_speed < angular_speed_limit:
        #     reward += 0.01

        # if abs(self.sim.pose[3]) < np.pi / 4:
        #     reward += 0.02

        # if abs(self.sim.pose[4]) < np.pi / 4:
        #     reward += 0.02

        # if (self.current_pos[2] < self.target_pos[2] and 0 < self.sim.v[2]) \
        #         or (self.target_pos[2] < self.current_pos[2] and self.sim.v[2] < 0):
        #     reward += 0.05

        # ============

        # reward = 0
        # current_distance = self.current_distance
        # future_pos = self.future_pos
        # future_distance = np.linalg.norm(self.target_pos - future_pos)
        # proximity = 2.0
        # perimeter = 5.0

        # speed = self.speed
        # speed_limit = 0.5

        # future_v = self.speed + self.sim.linear_accel * self.sim.dt

        # accel = self.sim.linear_accel
        # accel_amount = np.linalg.norm(accel)
        # accel_limit = 1.0

        # angular_speed = np.linalg.norm(self.sim.angular_v)
        # angular_speed_limit = 0.1

        # if self.sim.done:
        #     return self.sim.time / self.sim.runtime * 2 - 1

        # # if (cosine_similarity([self.sim.v], [self.target_pos - self.current_pos])[0][0] > 0.5):
        # #     reward += 0.075

        # # if (cosine_similarity([self.sim.v + self.sim.linear_accel * self.sim.dt], [future_pos - self.current_pos])[0][0] > 0.5):
        # #     reward += 0.05

        # # reward += 0.01

        # if self.current_distance <= perimeter:
        #     reward += 0.01

        # if self.current_distance <= proximity:
        #     reward += 0.02

        # if speed < speed_limit:
        #     reward += 0.005

        # if accel_amount < accel_limit:
        #     reward += 0.001

        # if angular_speed < angular_speed_limit:
        #     reward += 0.001

        # if 7 * np.pi / 4 < self.sim.pose[3] or self.sim.pose[3] < np.pi / 4:
        #     reward += 0.001

        # if 7 * np.pi / 4 < self.sim.pose[3] or self.sim.pose[4] < np.pi / 4:
        #     reward += 0.001

        # for i in range(3):
        #     if (self.current_pos[i] <= self.target_pos[i] and 0 <= self.sim.v[i]) \
        #             or (self.target_pos[i] <= self.current_pos[i] and self.sim.v[i] <= 0):
        #         reward += 0.02

        #     if (self.future_pos[i] <= self.target_pos[i] and 0 <= future_v[i]) \
        #             or (self.target_pos[i] <= self.future_pos[i] and future_v[i] <= 0):
        #         reward += 0.005

        # # n, v = self.init_pos - self.target_pos, self.current_pos - self.target_pos
        # # sway = np.linalg.norm(self.current_pos - (self.target_pos + np.dot(v, n) / np.dot(n, n) * n))

        # # if self.init_distance == 0:
        # #     sway = 0
        # # else:
        # #     u = self.target_pos - self.init_pos
        # #     v = self.current_pos - self.init_pos

        # #     sway = np.dot(np.cross(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), v)

        # # if sway < 1:
        # #     reward += 0.25

        # ============

        # reward = 0
        # current_distance = self.current_distance

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])

        #     reward += (-(2.5 / (1 + np.exp(-2 * ((time_left / self.sim.runtime)) )) - 1) + 0.5) * 15
        #     # reward += (-(2.5 / (1 + np.exp(-0.5 * ((current_distance / self.init_distance) - 2) )) - 1) + 0.5) * 2

        #     return reward

        # future_pos = self.future_pos

        # reward += 0.7

        # future_distance = np.linalg.norm(self.target_pos - self.future_pos)

        # reward += 0.25 * (2 / (1 + np.exp(-1.5 * (self.init_distance - current_distance) / self.init_distance)) - 1)
        # reward += 0.25 * (2 / (1 + np.exp(-1.5 * (self.init_distance - future_distance) / self.init_distance)) - 1)

        # # for i in range(3):
        # #     init_axis_distance = abs(self.target_pos[i] - self.init_pos[i])
        # #     current_axis_distance = abs(self.target_pos[i] - self.current_pos[i])
        # #     future_axis_distance = abs(self.target_pos[i] - future_pos[i])

        # #     reward += 0.3 * (2 / (1 + np.exp(-current_axis_distance / init_axis_distance)) - 1)
        # #     reward += 0.3 * (2 / (1 + np.exp(-(current_axis_distance - future_axis_distance) / (init_axis_distance / 2))) - 1)

        # speed_limit = 2.0
        # speed = self.speed
        # future_v = self.sim.v + self.sim.linear_accel * self.sim.dt
        # future_speed = np.linalg.norm(future_v)

        # if speed > speed_limit:
        #     reward += -0.25 * (2 / (1 + np.exp(-3 * ((speed - speed_limit) / speed_limit))) - 1)

        # if future_speed > speed_limit:
        #     reward += -0.25 * (2 / (1 + np.exp(-3 * ((future_speed - speed_limit) / speed_limit))) - 1)

        # ============

        # reward = 0

        # if self.sim.done:
        #     time_left = max([0, self.sim.runtime - self.sim.time])

        #     reward += (-(2.5 / (1 + np.exp(-5 * ((time_left / self.sim.runtime) - 0.2) )) - 1) + 0.5) * 15

        #     return reward

        # future_pos = self.future_pos

        # reward += 0.3

        # future_distance = np.linalg.norm(self.target_pos - self.future_pos)
        # current_distance = self.current_distance
        # future_v = self.sim.v + self.sim.linear_accel * self.sim.dt

        # reward += 0.5 * (2 / (1 + np.exp(-(current_distance - future_distance) / self.init_distance)) - 1)
        # reward += 0.5 * np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.current_distance * self.speed + 0.00001)

        # ============

        # if self.sim.done:
        #     # reward = -((self.sim.runtime - self.sim.time) / self.sim.runtime * 0.7 + (curr_dist - self.init_distance) / max([curr_dist, self.init_distance]) * 0.3) * 20
        #     reward = -(self.sim.time / self.sim.dt * 0.5) if self.sim.time < self.sim.runtime else 0

        # reward = -1 / (1 + np.exp(-5 * ((self.current_distance / self.init_distance * 2) - 1))) + 1
        # reward += self.sim.time / self.sim.runtime - 0.5

        # if self.sim.done:
        #     # reward = -((self.sim.runtime - self.sim.time) / self.sim.runtime * 0.7 + (curr_dist - self.init_distance) / max([curr_dist, self.init_distance]) * 0.3) * 20
        #     reward = -50.0 if self.sim.time < self.sim.runtime else 50.0
        #     reward += -min([0, self.sim.runtime - self.sim.time]) / self.sim.runtime * 25.0

        # if self.sim.done:
        #     reward = (-(self.sim.runtime - self.sim.time) / self.sim.runtime) * self.current_distance + ((self.init_distance - self.current_distance) / self.init_distance)

            # future_v = self.sim.v + self.sim.linear_accel
            # reward += np.dot(future_v, self.target_pos - self.current_pos) / (self.current_distance * np.linalg.norm(future_v)) * 0.5

        # reward += 0.5 * np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.current_distance * self.speed) if (self.current_distance > 0 and self.speed > 0) else 0 # reward on velocity's directional correctness + speed

        # reward = (self.init_distance - self.current_distance) + (self.current_distance - self.last_distance) + 0.25 * (np.linalg.norm(self.current_pos - self.last_pos)) - 1

        # reward += -1 if abs(self.sim.pose[3]) > 45 or abs(self.sim.pose[4]) > 45 else 0
        # reward += np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.current_distance * speed + 0.0001) * min([2., speed]) # reward on velocity's directional correctness + speed

        # reward -= self.current_distance / self.init_distance

        # reward += 3 if self.current_distance < self.last_distance else -2
        # reward += -2 if self.current_distance > 0 else 1

        # reward += (self.init_distance / (self.current_distance + self.init_distance)) - 1 # reward on closing the distance
        # reward += 0.25 if np.linalg.norm(self.last_pos - self.current_pos) > 0 else -0.25 # reward on moving

        # reward += (
        #     np.dot(self.current_pos - self.last_pos, self.target_pos - self.last_pos)
        #     / ((self.current_distance * np.linalg.norm(self.target_pos - self.last_pos)) + 0.0001) * 2.
        # ) - 1.

        # reward += (np.dot(self.sim.v, self.target_pos - self.current_pos) / (self.current_distance * np.linalg.norm(self.sim.v) + 0.0001) * 2) - 1 # reward on velocity's directional correctness
        # reward += 1 if self.current_distance < self.last_distance else -1

        # if self.current_distance < self.init_distance:
        # else:
        #     reward = -((self.current_distance * (1 + self.sim.time)) + (self.current_distance - self.last_distance)) # current distance + time it took to get there

        return self.clip(-1, reward, 1)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state_all = []
        self.rotor_speeds = rotor_speeds
        for _ in range(self.action_repeat):
            self.last_pos = self.current_pos
            self.last_distance = self.current_distance
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            state_all.append(self.current_state)

        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.current_state] * self.action_repeat)
        self.last_distance = self.init_distance
        return state
