"""Classes that signal to OpenFOAM how to output its fields and that load them."""
from itertools import count
import os
from pathlib import Path
import time
from typing import Tuple, Union, Optional

from shapely.geometry import Polygon, Point
import skimage.draw as draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from cases import Coordinates


class InternalCloudField(Coordinates):
    """A (velocity or pressure) field.

    Used to load the fields outputted by OpenFOAM.

    InternalCloudField holds a sequence of (x, y, z) spatial coordinates as
    well as corresponding field values at those points. For example, it could
    contain the pressure at a sequence of points.
    """

    _field: np.ndarray

    def __init__(self, field_vals, x_array, y_array, z_array=None):
        super().__init__(x_array, y_array, z_array=z_array)
        self._field = np.array(field_vals)
        assert self.x.shape[0] == self._field.shape[0]

    @classmethod
    def from_coords(cls, coords, field_vals):
        return cls(field_vals, coords.x, coords.y, coords.z)

    @classmethod
    def from_openfoam_array(cls, arr):
        x_array = arr[:, 0]
        y_array = arr[:, 1]
        z_array = arr[:, 2]
        field_vals = arr[:, 3:]
        return cls(field_vals, x_array, y_array, z_array)

    @property
    def number_of_points(self):
        return self._field.shape[0]

    @property
    def field_dimensions(self):
        return self._field.shape[1]

    def __getitem__(self, i):
        coords = super().__getitem__(i)
        field_at_point = self._field[i]
        return np.hstack((coords, field_at_point))


class CloudArray:
    """Discretizes space and loads fields using that discretization.

    Since the 2 tasks are related, they are the responsibility of a single
    class:
    1. discretizing space: creates a uniformly space mesh form which OpenFOAM
       samples the velocity and pressure fields.

       OpenFOAM then outputs those fields as a series of tuples similar to
       (x_coord, y_coord, z_coords, field_value)
    2. loading fields: loads those fields in a
    """
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Union[float, Tuple[float, float]]
    n_steps: int

    def __init__(self,
                 x_range=(-2, 2),
                 y_range=(-2, 2),
                 z_range=.5,
                 n_steps=64):

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.n_steps = n_steps

    def coords(self):
        """Computes the (x, y, z) coordinates of the cloud array."""
        xs = np.linspace(*self.x_range, num=self.n_steps)
        ys = np.linspace(*self.y_range, num=self.n_steps)
        zs = np.ones(self.n_steps * self.n_steps) * self.z_range

        xx, yy = np.meshgrid(xs, ys)
        xs = xx.ravel()
        ys = yy.ravel()

        return Coordinates(xs, ys, zs)

    def convert_icloud_field(self,
                             icloud_field: InternalCloudField) -> np.array:
        """Represents an InternalCloudField spatially in an array."""
        out = np.zeros(
            (icloud_field.field_dimensions, self.n_steps, self.n_steps))

        for point in icloud_field:
            i = self._x_coord(point[0])
            j = self._y_coord(point[1])
            out[:, i, j] = point[3:]

        return out

    def create_geom_array(self, geom_coords: Coordinates):
        """Represents a geometry spatially in an array.

        So a square geometry would be an array containing all 0s and a square
        block of 1s.
        """
        out = np.ones((self.n_steps, self.n_steps))
        geom_polygon = Polygon(geom_coords)

        for (x, y, _) in self.coords():
            point = Point((x, y))
            if geom_polygon.intersects(point):
                i = self._x_coord(x)
                j = self._y_coord(y)
                out[i, j] = 0

        return out

    def _x_coord(self, point):
        return self._ij_coordinate(point, self.x_range, self.n_steps)

    def _y_coord(self, point):
        return self._ij_coordinate(point, self.y_range, self.n_steps)

    @staticmethod
    def _ij_coordinate(point, its_range, n_steps):
        range_beginning, range_end = its_range
        range_len = range_end - range_beginning
        percent = (point - range_beginning) / range_len
        return int(np.round((n_steps - 1) * percent))


def test_coords_cloud_array():
    x_range = (0, 2)
    y_range = (0, 2)
    z_range = .5
    n_steps = 3
    cloudarr = CloudArray(x_range, y_range, z_range, n_steps)
    coords = cloudarr.coords()

    expected_coords_array = np.array([
        [0, 0, 0.5],
        [1, 0, 0.5],
        [2, 0, 0.5],
        [0, 1, 0.5],
        [1, 1, 0.5],
        [2, 1, 0.5],
        [0, 2, 0.5],
        [1, 2, 0.5],
        [2, 2, 0.5],
    ])

    assert np.allclose(coords.x, expected_coords_array[:, 0])
    assert np.allclose(coords.y, expected_coords_array[:, 1])
    assert np.allclose(coords.z, expected_coords_array[:, 2])
    assert np.allclose(coords.numpy(), expected_coords_array)


def test_reformat_cloud_array():
    x_range = (0, 2)
    y_range = (0, 2)
    z_range = .5
    n_steps = 3
    cloudarr = CloudArray(x_range, y_range, z_range, n_steps)

    scalar_openfoam_array = np.array([
        [2, 2, 0.5, 1],
        [1, 2, 0.5, 2],
        [0, 2, 0.5, 3],
        [2, 1, 0.5, 4],
        [1, 1, 0.5, 5],
        [0, 1, 0.5, 6],
        [2, 0, 0.5, 7],
        [1, 0, 0.5, 8],
        [0, 0, 0.5, 9],
    ])
    scalar_icloud = InternalCloudField.from_openfoam_array(
        scalar_openfoam_array)

    scalar_expected_output = np.array([
        [9, 6, 3],
        [8, 5, 2],
        [7, 4, 1],
    ]).reshape((1, 3, 3))

    scalar_computed_output = cloudarr.convert_icloud_field(scalar_icloud)

    assert np.allclose(scalar_expected_output, scalar_computed_output)

    vector_openfoam_array = np.hstack(
        [scalar_openfoam_array, -scalar_openfoam_array[:, -1].reshape(9, 1)])

    vector_icloud = InternalCloudField.from_openfoam_array(
        vector_openfoam_array)

    vector_expected_output = np.array([[
        [9., 6., 3.],
        [8., 5., 2.],
        [7., 4., 1.],
    ], [
        [-9., -6., -3.],
        [-8., -5., -2.],
        [-7., -4., -1.],
    ]])

    vector_computed_output = cloudarr.convert_icloud_field(vector_icloud)

    assert np.allclose(vector_computed_output, vector_expected_output)


def test_create_geom_array():
    raise NotImplementedError


def load_all_fields_from_cloud_array(
        cloud_array: CloudArray,
        dir_with_field_files='./OpenFOAM/postProcessing/internalCloud/150/',
        drop_z=True):
    """Loads fields outputted by OpenFOAM."""
    def load_field_from_cloud_array(
            cloud_array: CloudArray,
            filename='./OpenFOAM/postProcessing/internalCloud/150/cloud_U.xy',
    ):
        icloud_array = np.loadtxt(filename)
        icloud_field = InternalCloudField.from_openfoam_array(icloud_array)
        return cloud_array.convert_icloud_field(icloud_field)

    pressure_file = Path(dir_with_field_files) / 'cloud_p.xy'
    velocity_file = Path(dir_with_field_files) / 'cloud_U.xy'

    pressure_field = load_field_from_cloud_array(cloud_array, pressure_file)
    velocity_field = load_field_from_cloud_array(cloud_array, velocity_file)

    if drop_z:
        velocity_field = velocity_field[:2, :, :]

    speed_field = np.linalg.norm(velocity_field, axis=0)

    return {
        'pressure': pressure_field,
        'velocity': velocity_field,
        'speed': speed_field
    }


def save_scalar_field_as_image(filename, field):
    plt.imsave(filename, field)


def save_field_images(sample_dir, data_array):
    """Saves all field images."""
    speed_field = get_field_from_data_array(data_array, 'speed')
    pressure_field = get_field_from_data_array(data_array, 'pressure')

    speed_image_file = Path(sample_dir) / 'speed.png'
    pressure_image_file = Path(sample_dir) / 'pressure.png'

    save_scalar_field_as_image(speed_image_file, speed_field)
    save_scalar_field_as_image(pressure_image_file, pressure_field)


def load_forces(
        forces_file='./OpenFOAM/postProcessing/forceCoeffs_object/0/forceCoeffs.dat',
        return_all=False):
        """Load forces file outputted by OpenFOAM."""
    dat_array = np.loadtxt(forces_file)

    if return_all:
        return {
            'C_D': dat_array[:, 2],
            'C_L': dat_array[:, 3],
            'times': dat_array[:, 0]
        }

    final_forces_array = dat_array[-1]
    return {'C_D': final_forces_array[2], 'C_L': final_forces_array[3]}


def create_data_array(cloud_array: CloudArray,
                      openfoam_dir='./OpenFOAM') -> np.ndarray:
    """Creates sample data array from fields outputted by OpenFOAM."""
    openfoam_dir = Path(openfoam_dir)
    forces_file = './OpenFOAM/postProcessing/forceCoeffs_object/0/forceCoeffs.dat'
    internal_cloud_final = openfoam_dir / 'postProcessing/internalCloud/150/'

    fields = load_all_fields_from_cloud_array(
        cloud_array, dir_with_field_files=internal_cloud_final, drop_z=True)
    forces = load_forces(forces_file=forces_file)

    velocity_field = fields['velocity']
    pressure_field = fields['pressure']

    spatial_dimensions = pressure_field.shape[1:]
    shape_of_output = (4, ) + spatial_dimensions

    cd_field = np.ones(spatial_dimensions) * forces['C_D']
    merged = np.empty(shape_of_output)

    merged[:2, :, :] = velocity_field
    merged[2, :, :] = pressure_field
    merged[3, :, :] = cd_field

    return merged


def get_field_from_data_array(data_array: np.ndarray,
                              which_field='velocity') -> np.ndarray:
    """Gets a field from data array.

    The data array is created by create_data_array.
    The argument which_field should be one of 'velocity', 'speed', 'pressure'
    and 'C_D'.
    """
    if which_field in ('velocity', 'speed'):
        velocity_field = data_array[:2, ...]

    if which_field == 'velocity':
        return velocity_field
    if which_field == 'speed':
        return np.linalg.norm(velocity_field, axis=0)
    if which_field == 'pressure':
        return data_array[2, ...]
    if which_field == 'C_D':
        return data_array[3, ...]
