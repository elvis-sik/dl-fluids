"""Classes describing a flow case and functions to create one."""
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import List, Tuple, Iterable, NamedTuple, Optional, Union, Iterator

from shapely.geometry import Polygon, Point
import skimage.draw as draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


class Coordinates:
    """Spatial coordinates.


    If used to describe a geometry, it will be a polygon (or polyhedron) with
    vertices on the given coordinates. To instantiate this class, pass it a
    list of points and optionally a description of how they were obtained. The
    description is a string conventionally containing code that would produce
    the points (which would normally be a single function call):

    >>> x_array = np.array([0, 0, 1, 1])
    >>> y_array = np.array([0, 1, 0, 1])
    >>> description = 'np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])'
    >>> Coordinates(x_array, y_array, description=description)

    This class is useful because a (2D) geometry sometimes is better thought of
    as a 2 vectors (x and y coordinates) and sometimes as an array with 2
    columns.
    """
    _x_array: np.ndarray
    _y_array: np.ndarray
    _z_array: Optional[np.ndarray]

    def __init__(self, x_array, y_array, z_array=None, description=None):
        self._x_array = np.array(x_array).reshape(-1)
        self._y_array = np.array(y_array).reshape(-1)
        assert self._x_array.shape == self._y_array.shape

        if z_array is not None:
            self._z_array = np.array(z_array).reshape(-1)
            assert self._z_array.shape == self._x_array.shape
        else:
            self._z_array = None

        self.description = description

    @property
    def x(self):
        return self._x_array

    @property
    def y(self):
        return self._y_array

    @property
    def z(self):
        return self._z_array

    @property
    def is_2D(self):
        return self.z is None

    @property
    def is_3D(self):
        return not self.is_2D

    def __getitem__(self, i):
        point = [self.x[i], self.y[i]]
        if self.is_3D:
            point.append(self.z[i])
        return np.array(point)

    def __len__(self):
        return self.x.size

    def numpy(self):
        """Returns the coordinates as a matrix with  2 (or 3) columns."""
        row_list = [self.x, self.y]
        if self.is_3D:
            row_list.append(self.z)

        return np.array(row_list).T

    def rotate(self, angle):
        """Return new Coordinates object rotated clockwise.

        The angle parameters is in radians.
        """
        if self.description:
            description = f'{self.description}.rotate({angle})'
        else:
            description = None

        assert self.is_2D

        c, s = np.cos(angle), np.sin(angle)

        rotation_matrix = np.matrix([
            [c, s],
            [-s, c],
        ])
        points = np.array([self.x.reshape((1, -1)),
                           self.y.reshape((1, -1))]).reshape(2, -1)

        rotated = rotation_matrix @ points

        x_rotated = np.ravel(rotated[0, :])
        y_rotated = np.ravel(rotated[1, :])
        return Coordinates(x_rotated, y_rotated, description=description)


def plot_file(filename, skiprows=1):
    """Plot coordinates saved in text file.

    Mainly used for debugging.
    """
    points = np.loadtxt(filename, skiprows=skiprows)
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x, y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def _save_2d_dat(coords: Coordinates,
                 header='x y coordinates:',
                 filename='coordinates.dat'):
    """Saves Coordinates object in textual file."""
    def comment_out(text):
        """Returns lines of text with # prepended."""
        commented = []
        for line in text.splitlines():
            commented.append('# ' + line)
        return '\n'.join(commented)

    x, y = coords.x, coords.y
    assert x.shape == y.shape
    assert len(x.shape) == 1
    n = x.shape[0]

    header = comment_out(header)

    with open(filename, 'w') as file:
        indent = ' ' * 2
        separation = indent
        print(indent + header, file=file)
        for i in range(n):
            print(f'{indent}{x[i]}{separation}{y[i]}', file=file)


def circle_coordinates(radius: float = .5, n_steps: int = 150) -> Coordinates:
    """Returns Coordinates object of a circle.

    n_steps is the number of points used in the discretization.
    """
    description = f'circle_coordinates({radius}, {n_steps})'
    step_size = np.pi * 2 / n_steps

    x_list = []
    y_list = []

    for i in range(n_steps):
        t = step_size * i
        coord_x, coord_y = radius * np.array([np.cos(t), np.sin(t)])
        x_list.append(coord_x)
        y_list.append(coord_y)

    x = np.array(x_list, dtype='float')
    y = np.array(y_list, dtype='float')

    return Coordinates(x, y, description=description)


def create_circle_dat(radius: float = 1., filename: str = 'coordinates.dat'):
    """Saves circle coordinates to a text file."""
    coords = circle_coordinates(radius)
    header = 'x y coordinates of circle'
    _save_2d_dat(coords, header=header, filename=filename)


def polygon_coordinates(circ_radius=.5, vertices=4, rotation_angle=np.pi / 4):
    """Returns Coordinates object of a regular polygon.

    Arguments:
    + circ_radius: the radius of the circle from which the vertices are taken
    + vertices: number of vertices
    + rotation_angle: angle in radians through which the polygon is rotated
    """
    description = f'polygon_coordinates(circ_radius={circ_radius}, vertices={vertices}, rotation_angle={rotation_angle})'
    coords = circle_coordinates(circ_radius, vertices)
    rotated = coords.rotate(rotation_angle)
    rotated.description = description
    return rotated


def plot_points_with_order(x, y):
    """Plots sequence of points with numbers for debugging.

    The use case is to visually inspect if points which *should* be adjacent
    really are.
    """
    assert x.shape == y.shape
    assert len(x.shape) == 1 or x.shape

    for i in range(x.shape[0]):
        plt.scatter(x[i], y[i], marker=f'${i}$', color='blue')
    plt.show()
    plt.gca().set_aspect('equal', adjustable='box')


class SampleDirIter:
    """Creates a sequence of directories to be used for storing samples.

    >>> dir_iter = SampleDirIter(base_directory='./data/')
    >>> next_sample_dir = next(dir_iter)
    >>> next_sample_dir
    pathlib.Path('./data/1')
    >>> next_sample_dir.exists()
    True

    The class always looks for the smallest number to be filled. So if the
    100-th sample was removed by the solver because the solution
    ended in a mistake, SampleDirIter will create it again ('./data/100')
    """
    def __init__(self, base_directory: Union[Path, str]) -> Iterator[Path]:
        self.base_directory = Path(base_directory)

    def __iter__(self):
        return self

    def __next__(self):
        current_dir = self.base_directory / str(self.find_next_dir())
        current_dir.mkdir(parents=True)
        return current_dir

    def find_next_dir(self):
        """Finds the smallest number for which there is no sample directory."""
        training_data_dirs = []
        for path in self.base_directory.glob('*'):
            if path.is_dir():
                dir_number = int(path.name)
                training_data_dirs.append(dir_number)
        try:
            return max(training_data_dirs) + 1
        except ValueError:
            return 1


Velocity2D = Tuple[float, float]


@dataclass
class PhysParams:
    """Physical parameters of simulation."""
    reynolds: float
    freestream_vel: Velocity2D
    nu: float
    diameter: float

    # reference area for calculation of lift / drag coefficients
    a_ref: float

    # used for meshing, not 100% sure on what it does though
    edge_lc: float = 0.2

    def _asdict(self):
        # here for backwards compatibility purposes
        # PhysParams used to be a NamedTuple, which supports this method
        return asdict(self)


Case = NamedTuple('Case', geom_coords=Coordinates, physics=PhysParams)


def polygons_generator(n_cases=10,
                       min_reynolds=20,
                       max_reynolds=60,
                       min_vertices=3,
                       max_vertices=7,
                       nu=5e-2,
                       diameter=.5,
                       edge_lc=0.2):
    """Generates a sequence of polygon flow cases.

    The Reynolds number and number of vertices are both sampled uniformly from
    the ranges given. To understand the other parameters, see the docs for
    PhysParams.
    """
    for _ in range(n_cases):
        reynolds = np.random.uniform(min_reynolds, max_reynolds)
        vel_x = reynolds * nu / diameter
        physics = PhysParams(reynolds=reynolds,
                             freestream_vel=(vel_x, 0),
                             nu=nu,
                             a_ref=diameter * 2,
                             diameter=diameter,
                             edge_lc=edge_lc)

        rotation_angle = np.random.uniform(0, 2 * np.pi)
        vertices = np.random.choice(
            np.arange(min_vertices, max_vertices, dtype=int))
        geom_coords = polygon_coordinates(circ_radius=diameter,
                                          vertices=vertices,
                                          rotation_angle=rotation_angle)

        yield Case(geom_coords=geom_coords, physics=physics)
