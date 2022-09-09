"""grid_world.py
A simple grid world environment. Edges of the map are treated like obstacles
Map must be a image file whose values represent free (255, white), occupied (0, black).
"""

from __future__ import print_function, absolute_import, division

import cv2
import numpy as np
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import wrap_angles, compute_connected_pixels, rc_to_xy
from bc_exploration.utilities.util import xy_to_rc


class GridWorld:
    """
    A simple grid world environment. Edges of the map are treated like obstacles
    Map must be a image file whose values represent free (255, white), occupied (0, black).
    """

    def __init__(self,
                 map_filename,
                 map_resolution,
                 sensor,
                 footprint,
                 start_state=None,
                 map_padding=0.,
                 render_size=(500, 500)):
        """
        Creates an interactive grid world environment structured similar to open ai gym.
        Allows for moving, sensing, and visualizing within the space. Map loaded is based off the map_filename.
        :param map_filename str: path of image file whose values represent free (255, white), occupied (0, black).
        :param map_resolution float: resolution of which to represent the map
        :param sensor Sensor: sensor to use to sense the environment
        :param footprint Footprint: footprint of the robot
        :param start_state bool: if None, will randomly sample a free space point as the starting point
                                  else it must be [row, column] of the position you want the robot to start.
                                  it must be a valid location (free space and footprint must fit)
        :param map_padding float: padding in meters of which to pad the map with unexplored space
        :param render_size Tuple(int): size of which to render the map (for visualization env.render() )
        """

        self._footprint = footprint
        self._footprint_no_inflation = footprint.no_inflation()
        self._map_resolution = map_resolution
        self._map_padding = map_padding

        assert render_size.shape[0] == 2 if isinstance(render_size, np.ndarray) else len(render_size) == 2
        self._render_size = render_size

        self._state = None
        self._map = None
        self._truth_free_coords = None

        self._load_map(map_filename, map_resolution, map_padding)

        self._start_state = np.array(start_state).astype(np.float) \
            if start_state is not None else self.get_random_start_state()
        assert self._start_state.shape[0] == 3

        # shift start state to be [0., 0.]
        start_coord = xy_to_rc(self._start_state, self._map)
        self._map.origin = -self._start_state[:2]
        self._start_state = rc_to_xy(start_coord, self._map)

        self._sensor = sensor
        assert self._map is not None
        self._sensor.set_map(occupancy_map=self._map)

        self.reset()
        assert self._is_state_valid(self._start_state)

    def get_random_start_state(self):
        """
        Samples a random valid state from the map
        :return array(3)[float]: state [x, y, theta] of the robot
        """
        valid_points = rc_to_xy(np.argwhere(self._map.data == Costmap.FREE), self._map)
        choice = np.random.randint(valid_points.shape[0])

        valid = False
        while not valid:
            choice = np.random.randint(valid_points.shape[0])
            valid = self._is_state_valid(np.concatenate((valid_points[choice], [0])))

        # todo add random angle
        return np.concatenate((valid_points[choice], [0])).astype(np.float)

    def _is_state_valid(self, state, use_inflation=True):
        """
        Make sure state is not on an obstacle (footprint check)
        :param state array(3)[float]: [x, y, theta], the position/orientation of the robot
        :param use_inflation bool: whether to use the inflated footprint to collision check, or the normal footprint.
                              usually the environment will need use the actual footprint of the robot (because thats the
                              physical limitation we want to simulate)
        :return bool: whether it is valid or not
        """
        footprint = self._footprint if use_inflation else self._footprint_no_inflation
        return not footprint.check_for_collision(state=state, occupancy_map=self._map)

    def _load_map(self, filename, map_resolution, map_padding):
        """
        Loads map from file into costmap object
        :param filename str: location of the map file
        :param map_resolution float: desired resolution of the loaded map
        """
        map_data = cv2.imread(filename)
        assert map_data is not None and "map file not able to be loaded. Does the file exist?"
        map_data = cv2.cvtColor(map_data, cv2.COLOR_RGB2GRAY)
        map_data = map_data.astype(np.uint8)
        assert [value in [Costmap.FREE, Costmap.UNEXPLORED, Costmap.OCCUPIED] for value in np.unique(map_data)]
        map_data = np.pad(array=map_data, pad_width=int(np.rint(map_padding / map_resolution)), mode='constant', constant_values=Costmap.OCCUPIED)
        self._map = Costmap(data=map_data, resolution=map_resolution, origin=[0., 0.])

    def compare_maps(self, occupancy_map):
        """
        Does a comparison of the ground truth map with the input map,
        and will return a percentage completed
        :param occupancy_map Costmap: input map to be compared with ground truth
        :return float: percentage covered of the input map to the ground truth map
        """
        if self._truth_free_coords is None:
            start_state_px = xy_to_rc(self._start_state, self._map)
            self._truth_free_coords = compute_connected_pixels(start_state_px, self._map.data)

        free_coords = np.argwhere(occupancy_map.data == Costmap.FREE)
        return free_coords.shape[0] / float(self._truth_free_coords.shape[0])

    def step(self, desired_state):
        """
        Execute the given action with the robot in the environment, return the next robot position,
        and the output of sensor.measure() (sensor data)
        :param desired_state array(3)[float]: desired next state of the robot
        :return array(3)[float]: new_state (new robot position), sensor_data (output from sensor)
        """
        # sanitize input
        new_state = np.array(desired_state, dtype=np.float)
        new_state[2] = wrap_angles(desired_state[2])

        # if desired state is not valid, we dont move the robot
        if not self._is_state_valid(new_state, use_inflation=False):
            new_state = self._state.copy()

        # compute sensor_data
        sensor_data = self._sensor.measure(new_state)

        # set state
        self._state = new_state

        return new_state.copy(), sensor_data

    def reset(self):
        """
        Resets the robot to the starting state in the environment
        :return array(3)[float]: robot position, sensor data from start state
        """
        self._state = self._start_state.copy()
        sensor_data = self._sensor.measure(self._start_state)
        return self._state.copy(), sensor_data

    def render(self, wait_key=0):
        """
        Renders the environment and the robots position
        :param wait_key int: the opencv waitKey arg
        """
        # convert to colored image
        map_vis = cv2.cvtColor(self._map.data.copy(), cv2.COLOR_GRAY2BGR)

        state_px = xy_to_rc(self._state + self._start_state, self._map)[:2].astype(np.int)
        map_vis[state_px[0], state_px[1]] = [127, 122, 10]
        # # todo figure out programmatic way to pick circle size (that works well)
        # cv2.circle(map_vis, tuple(self.state[:2][::-1].astype(int)), 1, [127, 122, 10], thickness=-1)

        # resize map
        map_vis = cv2.resize(map_vis, tuple(self._render_size), interpolation=cv2.INTER_NEAREST)

        # visualize map
        cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('map', map_vis)
        cv2.resizeWindow('map', *self._render_size)
        cv2.waitKey(wait_key)

    def get_sensor(self):
        """
        Gets the sensor object
        :return Sensor: the sensor used by the grid world
        """
        return self._sensor

    def get_map_shape(self):
        """
        Returns the shape of the map
        :return Tuple(int): map shape
        """
        return self._map.get_shape()

    def get_map_size(self):
        """
        Returns the size of the map (x, y) in meters
        :return Tuple(float): map size (meters)
        """
        return self._map.get_size()

    def get_map_resolution(self):
        """
        Returns the resolution of the map
        :return float: resolution
        """
        return self._map_resolution

    def get_map_origin(self):
        return self._map.origin

    def get_start_state(self):
        return self._start_state

    def __del__(self):
        """
        Destructor, delete opencv windows
        """
        cv2.destroyAllWindows()
