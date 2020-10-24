"""Functions to process data after it has been created.

The idea is to use the funcs here to process data after solving cases the
following way:
1. manually creating several copies of the "solver" dir, all in the same
   directory with names like "solver_0", "solver_1" etc ("solver*" is the exact
   pattern)
2. let each of them solve cases while storing the data in one of their own
   subdirectories (e.g., "solver_0/data/")

Why do that:
     OpenFOAM uses a directory ("solver/OpenFOAM" here) to represent a single
     case so we at least need to create copies of it to solve different cases
     in parallel. It was just convenient to copy the whole "solver" dir and let
     it run in a separate process.

The problems the funcs here are solving:
+ if you for some reason interrupt your run (say, with "C-c"), you can end up
  with empty directories for some samples. find_all_problematic_cases helps
  with that
+ the data ends up spread around several directories.
"""
from itertools import chain
from pathlib import Path
from shutil import rmtree
from typing import Iterator


def get_all_data_dirs(dir_with_solvers='.',
                      data_subdir_name: str = 'data') -> Iterator[Path]:
    """Yields all "solver*/data" directories."""
    for directory in Path(dir_with_solvers).glob('solver*'):
        data_dir = directory / data_subdir_name
        if data_dir.exists():
            yield data_dir


def get_all_samples(dir_with_solvers='.', data_subdir_name: str = 'data'):
    """Generates all solver*/data/* sample directories."""
    for data_dir in get_all_data_dirs(dir_with_solvers, data_subdir_name):
        yield from data_dir.glob('*')


def check_sample_is_ok(sample_dir):
    """Checks whether a sample directory contains all expected files."""
    expected_files = frozenset((
        'data.npy',
        'forceCoeffs.dat',
        'geom.npy',
        'log.icoFoam',
        'params.json',
        'pressure.png',
        'speed.png',
    ))
    present_files = frozenset(pf.name for pf in sample_dir.glob('*'))

    return expected_files == present_files


def find_problematic_cases_single_dir(data_dir):
    """Yields all sample directories that don't have all expected files."""
    for sample_dir in Path(data_dir).glob('*'):
        if not check_sample_is_ok(sample_dir):
            yield sample_dir


def find_all_problematic_cases(dir_with_solvers='.', data_subdir_name='data'):
    """Yields all problematic samples across different solver* directories."""
    for data_dir in get_all_data_dirs(dir_with_solvers, data_subdir_name):
        yield from find_problematic_cases_single_dir(data_dir)


def remove_all_problematic_cases(dir_with_solvers='.',
                                 data_subdir_name='data'):
    for sample in find_all_problematic_cases():
        rmtree(sample)


def merge_data_dirs(target_dir, *data_dirs):
    """Moves all samples from the data dirs to a new one with orderered names."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    per_dir_sample_gens = (Path(dd).glob('*') for dd in data_dirs)
    sample_gen = chain(*per_dir_sample_gens)

    for i, sample_dir in enumerate(sample_gen, start=1):
        sample_dir.rename(target_dir / str(i))


def move_all_samples(target_dir='all-data',
                     dir_with_solvers='.',
                     data_subdir_name='data'):
    """Moves all samples from solver* dirs to a new one with ordered names."""
    data_dirs = get_all_data_dirs(dir_with_solvers, data_subdir_name)
    merge_data_dirs(target_dir, *data_dirs)


def split_data_dir(data_dir, n_train: int, n_val: int, n_test: int):
    """Split all data in a single dir in the 3 sets."""
    data_dir = Path(data_dir)

    all_samples = tuple(data_dir.glob('*'))
    assert len(all_samples) == n_train + n_val + n_test

    train_dir = data_dir / 'train'
    train_dir.mkdir(exist_ok=True)
    val_dir = data_dir / 'val'
    val_dir.mkdir(exist_ok=True)
    test_dir = data_dir / 'test'
    test_dir.mkdir(exist_ok=True)

    current_sample_index = 0

    all_n = n_train, n_val, n_test
    data_splits = train_dir, val_dir, test_dir

    for n, target in zip(all_n, data_splits):
        for i in range(1, n + 1):
            current_sample = all_samples[current_sample_index]
            current_sample.rename(target / str(i))
            current_sample_index += 1
