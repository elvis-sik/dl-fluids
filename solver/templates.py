"""Functions used to create OpenFOAM cases."""
from itertools import count
import os
from pathlib import Path
import time

from shapely.geometry import Polygon, Point
import skimage.draw as draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from .cases import Coordinates


def generate_mesh(coords: Coordinates, edge_lc=0.2, openfoam_dir='./OpenFOAM'):
    """Create mesh from coordinates using gmsh."""
    geo_str = _geo_file_template(coords, edge_lc)
    _generate_mesh_from_geostr(geo_str=geo_str, openfoam_dir=openfoam_dir)


def _geo_file_template(coords: Coordinates, edge_lc: float = .2) -> str:
    # if the first point is too close to the final point,
    # treat them as redumdant by eliminating the last one
    if np.linalg.norm(coords[0] - coords[-1]) < 1e-6:
        coords = coords[:-1]

    lines = []
    point_index = 1000 - 1
    for (x, y) in coords:
        point_index += 1
        lines.append(
            f"Point({point_index}) = {{ {x}, {y}, 0.00000000, 0.005}};")

    lines.append('\n')
    lines.append(f'Spline(1000) = {{1000:{point_index},1000}};')
    lines.append(f'edge_lc = {edge_lc};')

    rest = """
Point(1900) = { 5, 5, 0, edge_lc};
Point(1901) = { 5, -5, 0, edge_lc};
Point(1902) = { -5, -5, 0, edge_lc};
Point(1903) = { -5, 5, 0, edge_lc};

Line(1) = {1900,1901};
Line(2) = {1901,1902};
Line(3) = {1902,1903};
Line(4) = {1903,1900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {1000};
Plane Surface(1) = {1,2};

Extrude {0, 0, 1} {
  Surface{1};
  Layers{1};
  Recombine;
}
Physical Surface("back") = {1027};
Physical Surface("front") = {1};
Physical Surface("top") = {1022};
Physical Surface("exit") = {1010};
Physical Surface("bottom") = {1014};
Physical Surface("inlet") = {1018};
Physical Surface("aerofoil") = {1026};
Physical Volume("internal") = {1};
"""

    geo_str = '\n'.join(lines) + rest
    return geo_str


def _generate_mesh_from_geostr(geo_str: str, openfoam_dir='./OpenFOAM'):
    # Adapted from code available at
    # https://github.com/thunil/Deep-Flow-Prediction
    # Under the Apache license
    def save_geo_str(geo_str, filename='geometry.geo'):
        with open(filename, 'wt') as geo_file:
            geo_file.write(geo_str)

    def create_mesh():
        if os.system(
                "gmsh geometry.geo -format msh2 -3 -o geometry.msh > /dev/null"
        ) != 0:
            raise IOError("error during mesh creation!")

    def convert_mesh_to_openfoam():
        if os.system("gmshToFoam geometry.msh > /dev/null") != 0:
            raise IOError("error during conversion to OpenFoam mesh!")

    def correct_boundary():
        with open("constant/polyMesh/boundary", "rt") as in_file:
            with open("constant/polyMesh/boundaryTemp", "wt") as out_file:
                in_block = False
                in_aerofoil = False
                for line in in_file:
                    if "front" in line or "back" in line:
                        in_block = True
                    if in_block and "type" in line:
                        line = line.replace("patch", "empty")
                        in_block = False

                    if "aerofoil" in line:
                        in_aerofoil = True
                    if in_aerofoil and "type" in line:
                        line = line.replace("patch", "wall")
                        in_aerofoil = False
                    out_file.write(line)
        os.rename("constant/polyMesh/boundaryTemp",
                  "constant/polyMesh/boundary")

    previous_dir = Path.cwd()
    os.chdir(openfoam_dir)

    save_geo_str(geo_str)
    create_mesh()
    convert_mesh_to_openfoam()

    correct_boundary()

    os.chdir(previous_dir)


def create_internal_cloud_file(coords,
                               filename='./OpenFOAM/system/internalCloud'):
    """Creates the internalCloud file.

    This file indicates to OpenFOAM the coordinates of the points from which it
    should sample (and save) the values of the velocity and pressure fields.
    """
    assert coords.is_3D

    header = """/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Web:      www.OpenFOAM.org
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Writes out values of fields interpolated to a specified cloud of points.

\*---------------------------------------------------------------------------*/

fields (p U);
points
("""

    footer = """);

type            sets;
libs            ("libsampling.so");

interpolationScheme cellPoint;
setFormat	raw;

executeControl  writeTime;
writeControl    writeTime;

sets
(
    cloud
    {
        type    cloud;
        axis    xyz;
        points  $points;
    }
);

// ************************************************************************* //"""

    coords_array = coords.numpy()
    np.savetxt(filename,
               coords_array,
               fmt='(%f %f %f)',
               comments='',
               header=header,
               footer=footer)


def create_internal_cloud_dir(openfoam_dir='./OpenFOAM'):
    """Creates the directory in which OpenFOAM outputs the fields."""
    internal_cloud_dir = Path(
        openfoam_dir) / 'postProcessing' / 'internalCloud'
    internal_cloud_dir.mkdir(exist_ok=True, parents=True)


def create_boundary_conditions(freestream_vel, a_ref, openfoam_dir='.'):
    """Create file describing the boundary conditions of flow case."""
    freestream_x, freestream_y = freestream_vel

    openfoam_dir = Path(openfoam_dir)
    u_ref = np.sqrt(freestream_y**2 + freestream_x**2)

    with open(openfoam_dir / "U_template", "rt") as in_file:
        with open(openfoam_dir / "0/U", "wt") as out_file:
            for line in in_file:
                line = line.replace("VEL_X", str(freestream_x))
                line = line.replace("VEL_Y", str(freestream_y))
                out_file.write(line)

    with open(openfoam_dir / 'controlDict_template', 'rt') as in_file:
        with open(openfoam_dir / 'system/controlDict', 'wt') as out_file:
            for line in in_file:
                line = line.replace('A_REF', str(a_ref))
                line = line.replace('U_REF', str(u_ref))
                out_file.write(line)
