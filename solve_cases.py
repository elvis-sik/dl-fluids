"""Solve cases of flow simulation.

The 2 main functions here are:
+ solve_single_case: setup a flow case and call OpenFOAM to solve it
+ solve_cases: applies solve_single_case to a sequence of cases processing
               their output as a series of samples to be used by the neural
               network.
"""
from itertools import count
import json
import os
from pathlib import Path
import time
from typing import Iterator, List, Tuple, NamedTuple, Union
import sys

from shapely.geometry import Polygon, Point
import skimage.draw as draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

import solver.cloud_array as ca
import solver.cases as cases
from solver.templates import (generate_mesh, create_internal_cloud_file,
                       create_internal_cloud_dir, create_boundary_conditions)
import solver.validation as val


class SimulationError(BaseException):
    """Exception specific to OpenFOAM returning non-0 code"""
    pass


def run_simulation(openfoam_dir: Union[str, Path] = './solver/OpenFOAM/'):
    """Calls OpenFOAM on the case directory."""
    previous_dir = Path.cwd()
    os.chdir(openfoam_dir)

    exit_code = os.system('icoFoam > log.icoFoam')

    os.chdir(previous_dir)
    if exit_code != 0:
        raise SimulationError(f'OpenFOAM exited with code {exit_code}')


def move_files(sample_dir: Union[str, Path],
               openfoam_dir: Union[str, Path] = './solver/OpenFOAM'):
    """Moves files outputted by OpenFOAM to the sample directory."""
    def move(path, target_dir):
        """Moves file path to directory target_dir."""
        target_dir = Path(target_dir)
        path.rename(target_dir / path.name)

    openfoam_dir = Path(openfoam_dir)
    log_file = openfoam_dir / 'log.icoFoam'
    forces_file = openfoam_dir / 'postProcessing/forceCoeffs_object/0/forceCoeffs.dat'
    move(log_file, sample_dir)
    move(forces_file, sample_dir)


def clean_previous_run(openfoam_dir: Union[str, Path] = './solver/OpenFOAM',
                       verbose: bool = True):
    """Clears files outputted by a previous OpenFOAM run."""
    previous_dir = Path.cwd()
    os.chdir(openfoam_dir)
    if verbose:
        os.system('foamCleanTutorials')
    else:
        os.system('foamCleanTutorials > /dev/null')
    os.chdir(previous_dir)


def solve_single_case(geom_coords: cases.Coordinates,
                      physics: cases.PhysParams,
                      cloud_array: ca.CloudArray,
                      openfoam_dir: Union[str, Path] = './solver/OpenFOAM'):
    """Create and solve with OpenFOAM a flow case.

    Arguments:
    + geom_coords: geometry of the flow (e.g., the coordinates of a cilinder)
    + physics: parameters dealing with flow properties (such as rho) and meshing
    + cloud_array: see the docs for cloud_array.CloudArray
    + openfoam_dir: the OpenFOAM case directory
    """
    edge_lc = physics.edge_lc
    generate_mesh(geom_coords, openfoam_dir=openfoam_dir, edge_lc=edge_lc)

    icloud_filename = Path(openfoam_dir) / 'system/internalCloud'
    create_internal_cloud_file(cloud_array.coords(), filename=icloud_filename)
    create_internal_cloud_dir(openfoam_dir)

    create_boundary_conditions(physics.freestream_vel,
                               a_ref=physics.a_ref,
                               openfoam_dir=openfoam_dir)

    run_simulation(openfoam_dir=openfoam_dir)


def solve_cases(case_sequence: Iterator[cases.Case],
                cloud_array: ca.CloudArray,
                openfoam_dir='./solver/OpenFOAM',
                target_base_dir='./data/',
                verbose=True,
                n_cases=None,
                broken_cases_dir='./broken-cases'):
    """Solves a sequence of cases.

    Arguments:
    + case_sequence: a sequence of cases (see cases.Case)
    + cloud_array: see the docs for cloud_array.CloudArray
    + openfoam_dir: the OpenFOAM case directory
    + target_base_dir: the directory in which the samples will be stored
    + verbose: whether to output or not both errors and a count of cases
    + n_cases: the number of cases that will be solved. This is just used for
      printing. The caller should guarantee this will match the length of
      case_sequence.
    + broken_cases_dir: directory to which the cases that raise errors are
      moved. They can then be inspected for debugging.
    """
    def save_params(physics, geom_coords, sample_dir):
        """Creates json file describing the case in sample directory."""
        params = physics._asdict()

        params['geom_description'] = geom_coords.description

        with open(sample_dir / 'params.json', 'w') as file:
            json.dump(params, file)

    def save_geom_coords(geom_coords, sample_dir):
        """Saves an array containing the coordinates of the geometry."""
        geom_array = cloud_array.create_geom_array(geom_coords)
        np.save(sample_dir / 'geom.npy', geom_array)

    def move_broken_sample(sample_dir, broken_cases_dir):
        """Saves the files of a case that resulted in an error."""
        new_sample_dir = broken_cases_dir / sample_dir.name
        if new_sample_dir.exists():

            for i in count(start=1):
                new_sample_dir = broken_cases_dir / (sample_dir.name + f'_{i}')

                if not new_sample_dir.exists():
                    break

        sample_dir.rename(new_sample_dir)
        return new_sample_dir

    sample_dirs = cases.SampleDirIter(target_base_dir)
    broken_cases_dir = Path(broken_cases_dir).resolve()
    broken_cases_dir.mkdir(exist_ok=True, parents=True)

    for i, case, sample_dir in zip(count(1), case_sequence, sample_dirs):
        if verbose:
            if n_cases:
                print(f'{i} / {n_cases}')
            else:
                print(i)

        geom_coords = case.geom_coords
        physics = case.physics

        clean_previous_run(openfoam_dir, verbose=verbose)
        save_params(physics, geom_coords, sample_dir)
        save_geom_coords(geom_coords, sample_dir)

        try:
            solve_single_case(geom_coords,
                              physics,
                              cloud_array,
                              openfoam_dir=openfoam_dir)
        except SimulationError:
            sample_dir = move_broken_sample(sample_dir, broken_cases_dir)
        else:
            data_array = ca.create_data_array(cloud_array, openfoam_dir)
            np.save(sample_dir / 'data.npy', data_array)
            ca.save_field_images(sample_dir, data_array)
        finally:
            move_files(sample_dir, openfoam_dir)


if __name__ == '__main__':
    n_cases = 60
    polygon_cases = cases.polygons_generator(n_cases=n_cases,
                                             min_reynolds=75,
                                             max_reynolds=225,
                                             min_vertices=3,
                                             max_vertices=7,
                                             diameter=1.,
                                             nu=.01,
                                             edge_lc=2.)

    solve_cases(polygon_cases,
                openfoam_dir='./solver/OpenFOAM',
                cloud_array=ca.CloudArray(),
                verbose=True,
                n_cases=n_cases)
