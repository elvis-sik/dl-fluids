"""Datasets and neural network training.

See the factory functions at the end of the file:
>>> trainer = create_trainer()
>>> trainer.epochs(10)
"""
import abc
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import json
import itertools as it
from pathlib import Path
import time
from typing import Union, List, Dict, Any, Optional, Tuple, Callable, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from solver.cloud_array import load_forces
import unet

try:
    DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
except NameError:
    DEFAULT_CHECKPOINT_DIR = Path('./checkpoints')


class Sample(abc.ABC):
    sample_directory: Path
    _geom: Optional[np.ndarray]
    _data_array = Optional[np.ndarray]
    _params: Optional[Dict[Any, Any]]

    def __init__(self, sample_directory: Union[str, Path]) -> None:
        self.sample_directory = Path(sample_directory)
        self._geom = None
        self._data_array = None
        self._params = None

    @property
    def geom(self):
        if self._geom is None:
            self._geom = np.load(self.sample_directory / 'geom.npy')
        return self._geom

    @property
    def data_array(self):
        if self._data_array is None:
            self._data_array = np.load(self.sample_directory / 'data.npy')
        return self._data_array

    @property
    def params(self):
        if self._params is None:
            with open(self.sample_directory / 'params.json') as file:
                self._params = json.load(file)
        return self._params

    @property
    def reynolds(self):
        return self.params['reynolds']

    @property
    def freestream_vel(self):
        return self.params['freestream_vel']

    @property
    def freestream_speed(self):
        return np.linalg.norm(self.freestream_vel)

    @property
    def diameter(self):
        return self.params['diameter']

    def force_series(self, which_coeff='C_L'):
        force_coeffs_file = self.sample_directory / 'forceCoeffs.dat'
        forces_dict = load_forces(force_coeffs_file, return_all=True)
        return forces_dict[which_coeff]

    @abc.abstractmethod
    def nn_input(self):
        pass

    @abc.abstractmethod
    def nn_output(self):
        pass


class DefaultSample(Sample):
    def nn_input(self):
        nn_inp = np.array([self.geom, self.geom])
        nn_inp[0, ...] *= self.reynolds
        nn_inp[1, ...] *= self.freestream_vel[0]
        return nn_inp

    def nn_output(self):
        return self.data_array


class VelPressSample(DefaultSample):
    def nn_output(self):
        data_array = super().nn_output()
        without_cd = data_array[:3, ...]
        return without_cd


class NormalizedVelPressSample(VelPressSample):
    """Scale inputs / outputs.

    Output: a 3 x res x res array
    + Vel_out = velocity / scale_factor
    + P_out = (pressure - pressure_offset) / scale_factor
    """
    @classmethod
    def freeze_scaling_factors(cls, training_set_dir):
        factors = calculate_scaling_factors(training_set_dir)

        def sample_func(sample_directory):
            return cls(sample_directory, **factors)

        return sample_func

    def __init__(self, sample_directory: Union[str, Path], inp_vel_scale,
                 out_scale) -> None:
        super().__init__(sample_directory)
        self._inp_vel_scale = inp_vel_scale
        self._out_scale = out_scale

    def nn_input(self):
        nn_inp = super().nn_input()
        nn_inp[1, :, :] /= self._inp_vel_scale[0]
        return nn_inp

    def nn_output(self):
        nn_out = super().nn_output()
        nn_out[0, :, :] /= self._out_scale[0]
        nn_out[1, :, :] /= self._out_scale[1]
        nn_out[2, :, :] /= self._out_scale[2]
        nn_out[2, :, :] -= np.mean(nn_out[2, :, :])
        return nn_out

    def unnormalized_output(self):
        return super().nn_output()


class CDSample(NormalizedVelPressSample):
    def nn_output(self):
        # the layer 3 of the data_array contains the
        # drag coefficient C_D in all of its entries
        c_d = self.data_array[3, 0, 0]
        return np.ones((1, 1, 1), dtype=float) * c_d


def calculate_scaling_factors(training_set_dir):
    dataset = Dataset(training_set_dir, sample_type=VelPressSample)
    assert len(dataset) > 0

    inp_entries = []
    out_entries = []

    for sample in dataset:
        # sample_dict = dataset[0]
        sample_inp = sample['input'].numpy()
        sample_out = sample['output'].numpy()

        inp_entries.append(np.mean(np.abs(sample_inp[1, :, :])))
        out_entries.append(np.mean(np.abs(sample_out), axis=(1, 2)))

    inp_vel_scale = np.mean(np.vstack(inp_entries), axis=0)
    out_scale = np.mean(np.vstack(out_entries), axis=0)

    inp_vel_scale[np.isclose(inp_vel_scale, 0)] = 1
    out_scale[np.isclose(out_scale, 0)] = 1

    return {'inp_vel_scale': inp_vel_scale, 'out_scale': out_scale}


class ThuereyDataset(torch.utils.data.Dataset):
    def __init__(self, base_directory='./data'):
        self.base_directory = Path(base_directory)
        self.files = list(self.base_directory.glob('*.npz'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, sample_number):
        arr = np.load(self.files[sample_number])
        assert arr.files == ['a']
        arr = arr['a']

        inp = torch.from_numpy(arr[0:3])
        out = torch.from_numpy(arr[3:6])
        return {'input': inp, 'output': out}


class Dataset(torch.utils.data.Dataset):
    base_directory: Path
    samples: List[Path]
    _sample_type: Sample

    def __init__(self,
                 base_directory: Union[str, Path] = './data/',
                 sample_type: Sample = DefaultSample):
        self.base_directory = Path(base_directory)
        self._sample_type = sample_type
        self.samples = []

        for path in self.base_directory.glob('*'):
            if path.is_dir():
                self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_number):
        sample = self.get_sample(sample_number)

        return {
            'input': self._get_nn_input(sample),
            'output': self._get_nn_output(sample),
        }

    def get_sample(self, sample_number):
        sample_dir = self._get_sample_directory(sample_number)
        return self._create_sample(sample_dir)

    def _create_sample(self, sample_directory):
        return self._sample_type(sample_directory)

    def _get_sample_directory(self, sample_number):
        return self.samples[sample_number]

    def _get_nn_input(self, sample):
        return self._numpy_to_tensor(
            self._convert_array_to_float(sample.nn_input()))

    def _get_nn_output(self, sample):
        return self._numpy_to_tensor(
            self._convert_array_to_float(sample.nn_output()))

    def _numpy_to_tensor(self, array):
        tensor = torch.from_numpy(array)
        return tensor

    def _convert_array_to_float(self, array):
        # return array.astype('double')
        return array.astype(float)


class IdentityDataset:
    """Identity mapping dataset used for debugging."""
    def __init__(self,
                 num_samples=30,
                 shape=(3, 64, 64),
                 noise_std=10,
                 max_val=100):
        self.num_samples = num_samples
        self._shape = shape
        self._max_val = max_val
        self._noise = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._verify_idx_in_bounds(idx)
        return self._create_random_sample()

    def _create_random_sample(self):
        inp = self._random_tensor()
        out = self._add_noise(inp)
        return {'input': inp, 'output': out}

    def _random_tensor(self):
        tensor = self._max_val * torch.from_numpy(
            np.random.uniform(size=self._shape))
        return tensor

    def _add_noise(self, tensor):
        with torch.no_grad():
            return tensor + self._create_noise()

    def _create_noise(self):
        return self._noise * torch.rand(self._shape)

    def _verify_idx_in_bounds(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f'tried to get the {idx}-th sample of dataset of length {self.num_samples}'
            )


@dataclass
class TrainingProblem:
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    training_data: Dataset
    validation_data: Dataset
    test_data: Optional[Dataset]
    device: torch.DeviceObjType = torch.device('cpu')


def run_single_epoch(network,
                     optimizer,
                     loss_fn,
                     training_data,
                     lr_scheduler=None,
                     device='cpu'):
    """Run single epoch."""
    for data_dict in training_data:
        inp = data_dict['input']
        exp = data_dict['output']
        inp = inp.to(torch.float32)
        exp = exp.to(torch.float32)
        inp = inp.to(device)
        exp = exp.to(device)

        optimizer.zero_grad()
        out = network(inp)
        loss = loss_fn(out, exp)
        loss.backward()
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()


class LossRegistry:
    def __init__(self):
        self.epoch_number = 0
        self._training_loss_history = []
        self._validation_loss_history = []
        self.epochs_of_history = []

    def incr_epoch_number(self, n=1):
        self.epoch_number += 1

    def register_loss(self, training_loss, validation_loss):
        self.epochs_of_history.append(self.epoch_number)
        self._training_loss_history.append(training_loss)
        self._validation_loss_history.append(validation_loss)

    def last_loss(self, split='train'):
        return self._get_loss_history_by_split(split)[-1]

    def get_loss_history(self, split='train'):
        return tuple(self._get_loss_history_by_split(split))

    def _get_loss_history_by_split(self, split='train'):
        if split == 'train':
            return self._training_loss_history
        elif split == 'val':
            return self._validation_loss_history

    def params_for_saving(self):
        return {
            'epoch_number': self.epoch_number,
            'training_loss_history': self._training_loss_history,
            'validation_loss_history': self._validation_loss_history,
            'epochs_of_history': self.epochs_of_history
        }

    def load_params(self, *, epoch_number, training_loss_history,
                    validation_loss_history, epochs_of_history):
        self.epoch_number = epoch_number
        self._training_loss_history = training_loss_history
        self._validation_loss_history = validation_loss_history
        self.epochs_of_history = epochs_of_history


def relative_error_loss_maximum(pred, exp, aggregation=torch.mean):
    error_tensor = torch.abs(pred - exp) / torch.maximum(
        torch.abs(pred), torch.abs(exp))
    return aggregation(error_tensor)


def compute_single_loss(network,
                        loss_fn,
                        inp,
                        exp,
                        device='cpu',
                        zero_geom=True):
    inp = inp.to(torch.float32)
    exp = exp.to(torch.float32)
    inp = inp.to(device)
    exp = exp.to(device)

    with torch.no_grad():
        out = network(inp)

    if zero_geom:
        n_channels = out.shape[0]
        mask = inp[0] == 0
        for i in range(n_channels):
            out[i][mask] = 0

    return loss_fn(out, exp)


def compute_all_losses(network,
                       loss_fn,
                       data_gen,
                       device='cpu',
                       zero_geom=True):
    losses = []

    for data_dict in data_gen:
        inp = data_dict['input']
        exp = data_dict['output']
        losses.append(
            compute_single_loss(network,
                                loss_fn,
                                inp,
                                exp,
                                device=device,
                                zero_geom=zero_geom))

    return losses


def compute_average_loss(network,
                         loss_fn,
                         data_gen,
                         device='cpu',
                         zero_geom=True):
    total_loss = 0

    for data_dict in data_gen:
        inp = data_dict['input']
        exp = data_dict['output']
        total_loss += compute_single_loss(network,
                                          loss_fn,
                                          inp,
                                          exp,
                                          device=device,
                                          zero_geom=zero_geom)

    return float(total_loss / len(data_gen))


class Checkpointer(abc.ABC):
    checkpoint_dir: Path
    every: int

    def __init__(self, checkpoint_dir: Union[str, Path], every: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every = every
        self._next_in = every

    def checkpoint(self):
        self._next_in -= 1
        if self._next_in == 0:
            self._checkpoint()
            self._next_in = self.every

    def keyboard_interrupt(self):
        self._checkpoint()

    @abc.abstractmethod
    def _checkpoint(self):
        pass


class CombineCheckpointer(Checkpointer):
    _checkpointers: Tuple[Checkpointer]

    def __init__(self, checkpoint_dir, checkpointers=()):
        super().__init__(checkpoint_dir, 1)
        self._checkpointers = tuple(checkpointers)

    def _checkpoint(self):
        for checkpointer in self._checkpointers:
            checkpointer.checkpoint()

    def keyboard_interrupt(self):
        for checkpointer in self._checkpointers:
            checkpointer.keyboard_interrupt()


class EpochNumberCheckpointer(Checkpointer):
    def __init__(self, checkpoint_dir, loss_registry: LossRegistry):
        super().__init__(checkpoint_dir, 1)
        self._loss_registry = loss_registry

    def _checkpoint(self):
        self._loss_registry.epoch_number += 1


class LossCheckpointer(Checkpointer):
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 every: int,
                 loss_registry: LossRegistry,
                 training_problem: TrainingProblem,
                 zero_geom: bool = True):
        super().__init__(checkpoint_dir, every)
        self._training_problem = training_problem
        self._loss_registry = loss_registry
        self._zero_geom = zero_geom

    def _checkpoint(self):
        training_loss = self._average_loss(self._get_data('train'))
        validation_loss = self._average_loss(self._get_data('val'))
        self._loss_registry.register_loss(training_loss, validation_loss)

    def _get_data(self, split='train'):
        if split == 'train':
            return self._training_problem.training_data
        elif split == 'val':
            return self._training_problem.validation_data

    def _get_loss_history(self, split='val'):
        return self._loss_registry.get_loss_history(split)

    def _average_loss(self, data):
        return compute_average_loss(self._training_problem.network,
                                    self._training_problem.loss_fn,
                                    data,
                                    device=self.device,
                                    self._zero_geom)

    @property
    def device(self):
        return self._training_problem.device


class ModelManager:
    _OLD_CHECKPOINTS_DIRNAME = 'old'

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)

    def move_current_model(self):
        self._mkdir()
        old_model_file = self.current_model()

        if old_model_file is not None:
            self._move_to_checkpoint_dir(old_model_file)

    def current_model(self) -> Optional[Path]:
        nn_files = tuple(self.checkpoint_dir.glob('*.nn'))
        if not nn_files:
            return

        assert len(nn_files) == 1
        return nn_files[0]

    def build_filename(self, epoch_number, train_loss, val_loss):
        filename = f'Epoch {epoch_number} - Loss: train={train_loss}, val={val_loss}.nn'
        return self.checkpoint_dir / filename

    def _move_to_checkpoint_dir(self, path):
        path.rename(self._old_checkpoints_path() / path.name)

    def _mkdir(self):
        Path(self._old_checkpoints_path()).mkdir(exist_ok=True, parents=True)

    def _old_checkpoints_path(self):
        return self.checkpoint_dir / self._OLD_CHECKPOINTS_DIRNAME


class StateCheckpointer(Checkpointer):
    def __init__(self, checkpoint_dir: Union[str, Path], every,
                 training_problem: TrainingProblem,
                 model_manager: ModelManager, loss_registry: LossRegistry):
        super().__init__(checkpoint_dir, every)
        self._training_problem = training_problem
        self._manager = model_manager
        self._loss_registry = loss_registry

    def _checkpoint(self):
        assert self._loss_fn
        self._get_rid_of_current()
        self._save_nn(self._build_params_dict(), self._current_filename())

    def load_last_model(self):
        current_model = self._get_current_model()
        self.load_model(current_model)

    def load_model(self, filename):
        """Load saved checkpoint."""
        checkpoint_dict = self._load_nn(filename)
        self._torch_set_state(checkpoint_dict)
        self._set_loss_fn(checkpoint_dict)
        self._set_loss_registry_state(checkpoint_dict)

    def _set_loss_registry_state(self, checkpoint_dict: dict):
        loss_registry_params = checkpoint_dict['loss_registry']
        self._loss_registry.load_params(**loss_registry_params)

    def _set_loss_fn(self, checkpoint_dict):
        self._loss_fn = checkpoint_dict['loss_fn']

    def _torch_set_state(self, checkpoint_dict):
        net_state_dict = checkpoint_dict['network_state']
        opt_state_dict = checkpoint_dict['optimizer_state']

        self._network.load_state_dict(net_state_dict)
        self._optimizer.load_state_dict(opt_state_dict)

    def _current_filename(self):
        return self._manager.build_filename(
            self._epoch_number,
            self._last_loss('train'),
            self._last_loss('val'),
        )

    def _load_nn(self, filename):
        return torch.load(filename, map_location=self._training_problem.device)

    @staticmethod
    def _save_nn(params_dict, filename):
        torch.save(params_dict, filename)

    def _build_params_dict(self):
        return {
            'network_state': self._network_state(),
            'optimizer_state': self._optimizer_state(),
            'loss_fn': self._loss_fn,
            'loss_registry': self._loss_registry_params()
        }

    def _loss_registry_params(self):
        return self._loss_registry.params_for_saving()

    def _get_rid_of_current(self):
        self._manager.move_current_model()

    def _last_loss(self, split='train'):
        return self._loss_registry.last_loss(split)

    def _optimizer_state(self):
        return self._training_problem.optimizer.state_dict()

    def _network_state(self):
        return self._training_problem.network.state_dict()

    def _get_current_model(self):
        return self._manager.current_model()

    @property
    def _network(self):
        return self._training_problem.network

    @property
    def _optimizer(self):
        return self._training_problem.optimizer

    @property
    def _loss_fn(self):
        return self._training_problem.loss_fn

    @_loss_fn.setter
    def _loss_fn(self, new_loss_fn):
        self._training_problem.loss_fn = new_loss_fn

    @property
    def _epoch_number(self):
        return self._loss_registry.epoch_number


class PlotCheckpointer(Checkpointer):
    PLOT_FILE_NAME = 'loss-plot.png'

    def __init__(self, checkpoint_dir, every, loss_registry: LossRegistry):
        super().__init__(checkpoint_dir, every)
        self._loss_registry = loss_registry

    def _checkpoint(self):
        plt.clf()
        plt.plot(self._epochs_for_plot(),
                 self._loss_history('train'),
                 'r-',
                 label='training error')
        plt.plot(self._epochs_for_plot(),
                 self._loss_history('val'),
                 'b-',
                 label='validation error')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss history')
        plt.legend()
        plt.grid()
        plt.savefig(self.checkpoint_dir / self.PLOT_FILE_NAME)

    def _loss_history(self, split='train'):
        return self._loss_registry.get_loss_history(split)

    def _epochs_for_plot(self):
        return self._loss_registry.epochs_of_history


class Timer:
    def __init__(self):
        self._initial_time = None

    def start(self):
        assert self._initial_time is None
        self._initial_time = time.time()

    def time(self):
        elapsed_time = time.time() - self._initial_time
        return timedelta(seconds=elapsed_time)


class StdoutCheckpointer(Checkpointer):
    def __init__(self,
                 checkpoint_dir,
                 every,
                 loss_registry: LossRegistry,
                 timer: Optional[Timer] = None,
                 end_character='\r'):
        super().__init__(checkpoint_dir, every)
        self._loss_registry = loss_registry
        self._timer = timer
        self._end_character = end_character

    def _checkpoint(self):
        epoch = self._epoch_number()
        train_loss = self._current_loss('train')
        val_loss = self._current_loss('val')
        msg = f'Epoch: {epoch}, train loss: {train_loss}, val loss:: {val_loss}'
        if self._timer:
            elaps = self._elapsed_time()
            msg = msg + f' , elapsed time: {elaps}'

        print(msg, end=self._end_character)

    def _elapsed_time(self):
        return self._timer.time()

    def _current_loss(self, split):
        return self._loss_registry.last_loss(split)

    def _epoch_number(self):
        return self._loss_registry.epoch_number


class Trainer:
    problem: TrainingProblem
    checkpointer: Checkpointer

    def __init__(self, problem: TrainingProblem, checkpointer: Checkpointer):
        self.problem = problem
        self.checkpointer = checkpointer

    def epoch(self):
        self._run_epoch()
        self._checkpoint()

    def epochs(self, num_epochs=None):
        if num_epochs is None:
            iterator = it.repeat(None)
        else:
            iterator = range(num_epochs)

        try:
            for _ in iterator:
                self.epoch()
        except KeyboardInterrupt:
            self._checkpoint()

    def _run_epoch(self):
        run_single_epoch(self.problem.network,
                         self.problem.optimizer,
                         self.problem.loss_fn,
                         self.problem.training_data,
                         device=self.problem.device)

    def _checkpoint(self):
        self.checkpointer.checkpoint()


class Analyzer:
    def __init__(self, trainer, device='cuda', dtype=torch.float32):
        self.trainer = trainer
        self.device = device
        self.dtype = dtype
        self._relative_error_stats = {'train': None, 'val': None}
        self._relative_error_thuerey_memoized = {'train': None, 'val': None}

    @property
    def problem(self):
        return self.trainer.problem

    @property
    def training_data(self):
        return self.problem.training_data

    @property
    def validation_data(self):
        return self.problem.validation_data

    @property
    def test_data(self):
        return self.problem.test_data

    @property
    def loss_fn(self):
        return self.problem.loss_fn

    @property
    def network(self):
        return self.problem.network

    def apply_network(self,
                      i,
                      split='train',
                      return_type='torch',
                      squeeze=False):
        inp = self.ith_input(i, split).unsqueeze(0)

        with torch.no_grad():
            tens = self.network(inp)

        if squeeze:
            tens = tens.squeeze()

        if return_type == 'torch':
            return tens
        return tens.cpu().numpy()

    def _get_dataset(self, split):
        if split == 'train':
            return self.training_data.dataset
        elif split == 'val':
            return self.validation_data.dataset
        elif split == 'test':
            return self.test_data.dataset

    def ith_sample(self, i, split='train'):
        sample_dict = self._get_dataset(split)[i]
        out_dict = {}
        for key, val in sample_dict.items():
            tens = val.to(self.dtype)
            tens = tens.to(self.device)
            out_dict[key] = tens
        return out_dict

    def ith_input(self, i, split='train', return_type='torch'):
        arr = self.ith_sample(i, split=split)['input']
        if return_type == 'torch':
            return arr.to(self.device)
        return arr.cpu().numpy()

    def ith_exp(self, i, split='train', return_type='torch'):
        tens = self.ith_sample(i, split=split)['output']
        if return_type == 'torch':
            return tens
        return tens.cpu().numpy()

    def _velocity_component(self, out_arr, direction='speed'):
        vel_arr = out_arr[:2, :, :]

        if direction == 'speed':
            return np.linalg.norm(vel_arr, axis=0)
        elif direction == 'x':
            return vel_arr[0, :, :]
        elif direction == 'y':
            return vel_arr[1, :, :]

    def _zero_inside_mask(self, arr, i, split='train'):
        assert arr.ndim in {2, 3}

        inp = self.ith_input(i, split, return_type='numpy')
        mask = inp[0] == 0
        if arr.ndim == 2:
            arr[mask] = 0
        else:
            arr[:, mask] = 0
        return arr

    def plot_ith_vel_exp(self, i, direction='speed', split='train', axis=None):
        out_arr = self.ith_exp(i, split).cpu().numpy()
        out_arr = self._velocity_component(out_arr, direction)
        self._zero_inside_mask(out_arr, i, split)
        self._plot(out_arr, axis=axis)

    def plot_ith_vel_pred(self, i, direction='speed', split='train',
                          axis=None):
        minibatch_out_arr = self.apply_network(i, split).cpu().numpy()
        out_arr = minibatch_out_arr[0, :, :, :]
        out_arr = self._velocity_component(out_arr, direction)
        self._zero_inside_mask(out_arr, i, split)
        self._plot(out_arr, axis=axis)

    def plot_ith_press_exp(self, i, split='train', axis=None):
        arr = self.ith_exp(i, split)[2, :, :].cpu().numpy()
        self._zero_inside_mask(arr, i, split)
        self._plot(arr, axis=axis)

    def plot_ith_press_pred(self, i, split='train', axis=None):
        arr = self.apply_network(i, split)[0, 2, :, :].cpu().numpy()
        self._zero_inside_mask(arr, i, split)
        self._plot(arr, axis=axis)

    def _plot(self, arr, axis=None):
        if axis is None:
            plt.imshow(arr)
        else:
            axis.imshow(arr)

    def loss_on_ith_sample(self, i, split='train', channel='all'):
        exp = self.ith_exp(i, split)
        exp_batch = exp.unsqueeze(0)
        pred = self.apply_network(i, split)

        if channel == 'x':
            pred = pred[0, 0, :, :]
            exp_batch = exp_batch[0, 0, :, :]
        elif channel == 'y':
            pred = pred[0, 1, :, :]
            exp_batch = exp_batch[0, 1, :, :]
        elif channel == 'p':
            pred = pred[0, 2, :, :]
            exp_batch = exp_batch[0, 2, :, :]

        return float(self.loss_fn(pred, exp_batch))

    def all_losses(self, split='train', channel='all'):
        return [
            self.loss_on_ith_sample(i, split, channel)
            for i in range(self.num_samples(split))
        ]

    def num_samples(self, split='train'):
        return len(self._get_dataset(split))

    def relative_error_guo(self, split='train'):
        def compute_relative_error(output, target):
            inp = self.ith_input(i, split, return_type='numpy')
            mask = inp[0] == 0

            # drop pressure channel
            output = output[0:2, :, :]
            target = target[0:2, :, :]

            denom = np.linalg.norm(target, axis=0)
            denom[denom == 0] = 1

            error = np.linalg.norm(output - target, axis=0) / denom

            error[mask] = 0

            cells_inside_mask = np.sum(mask)
            cells_outside_mask = mask.size - cells_inside_mask

            return np.sum(error) / cells_outside_mask

        errors = []

        for i in range(self.num_samples(split)):
            pred = self.apply_network(i,
                                      split,
                                      return_type='numpy',
                                      squeeze=True)
            exp = self.ith_exp(i, split, return_type='numpy')

            errors.append(compute_relative_error(pred, exp))

        return errors

    def relative_error(self, i=None, channel='x', split='train', stat='all'):
        """Return relative error between expected and predicted output.

        channel must be one of 'x', 'y', 'p' or 'all' (x and y components of
        the velocity and pressure).

        stat must be one of 'mean', 'median', 'max', 'std', 'all' (which
        returns a dict with all of the previous) or 'array' (which returns a
        numpy array with the relative difference between expected and predicted
        output for each component of the chosen channels).
        """
        def build_channel_stat_dict(old_stat_dict, channel):
            new_stat_dict = {}
            for key, val in old_stat_dict.items():
                new_stat_dict[key] = val[channel]
            return new_stat_dict

        if channel == 'x':
            channel = 0
        elif channel == 'y':
            channel = 1
        elif channel == 'p':
            channel = 2

        if stat == 'array':
            if channel == 'all':
                channel = slice(0, 3)

            assert i is not None
            return self._relative_error_array(i, split, channel)

        self._compute_relative_error_stats(split)

        if channel == 'all':
            stat_list = self._relative_error_stats[split]['whole_sample']
            if i is None:
                if stat == 'all':
                    return stat_list
                return [sd[stat] for sd in stat_list]

            if stat == 'all':
                return stat_list[i]
            return stat_list[i][stat]

        stat_list = self._relative_error_stats[split]['per_channel']

        if i is None:
            if stat == 'all':
                return [
                    build_channel_stat_dict(sd, channel) for sd in stat_list
                ]

            return [sd[stat][channel] for sd in stat_list]

        if stat == 'all':
            return build_channel_stat_dict(stat_list[i], channel)

        return stat_list[i][stat][channel]

    def _compute_relative_error_stats(self, split='train'):
        def create_stat_dict(rel_err, axis):
            return {
                'mean': np.mean(rel_err, axis=axis),
                'median': np.median(rel_err, axis=axis),
                'max': np.max(rel_err, axis=axis),
                'std': np.std(rel_err, axis=axis),
            }

        if self._relative_error_stats[split] is not None:
            return

        per_channel_stat_dicts = []
        all_channels_stat_dicts = []

        for sample in range(self.num_samples(split)):
            rel_err_array = self._relative_error_array(sample, split)

            per_channel_stat_dicts.append(
                create_stat_dict(rel_err_array, (1, 2)))
            all_channels_stat_dicts.append(
                create_stat_dict(rel_err_array, None))

        self._relative_error_stats[split] = {
            'per_channel': per_channel_stat_dicts,
            'whole_sample': all_channels_stat_dicts
        }

    def _relative_error_array(self, i, split, mask_out_geom=True):
        exp = self.ith_exp(i, split, return_type='numpy')
        pred = self.apply_network(i, split, return_type='numpy', squeeze=True)
        if mask_out_geom:
            inp = self.ith_input(i, split, return_type='numpy')
            mask = inp[0] == 0

        # https://stats.stackexchange.com/a/201864
        error = np.abs(pred - exp) / np.maximum(np.abs(pred), np.abs(exp))
        for i in range(3):
            error[i][mask] = 0
        return error

    def relative_error_thuerey(self, i=None, channel='x', split='train'):
        self._build_relative_error_thuerey_list(split)

        if i is None:
            return [
                dct[channel]
                for dct in self._relative_error_thuerey_memoized[split]
            ]

        return self._relative_error_thuerey[split][i][channel]

    def _build_relative_error_thuerey_list(self, split):
        if self._relative_error_thuerey_memoized[split] is not None:
            return

        dict_list = []
        for i in range(self.num_samples(split)):
            dict_list.append(self._relative_error_thuerey(i, split))

        self._relative_error_thuerey_memoized[split] = dict_list

    def _relative_error_thuerey(self, i, split):
        exp = self.ith_exp(i, split, return_type='numpy')
        pred = self.apply_network(i, split, return_type='numpy', squeeze=True)

        return {
            'v':
            np.sum(
                np.abs(pred[0, :, :] - exp[0, :, :]) +
                np.abs(pred[1, :, :] - exp[1, :, :])) /
            (np.sum(np.abs(exp[0, :, :])) + np.sum(np.abs(exp[1, :, :]))),
            'all':
            np.sum(np.abs(pred - exp)) / np.sum(np.abs(exp)),
            'x':
            np.sum(np.abs(pred[0] - exp[0])) / np.sum(np.abs(exp[0])),
            'y':
            np.sum(np.abs(pred[1] - exp[1])) / np.sum(np.abs(exp[1])),
            'p':
            np.sum(np.abs(pred[2] - exp[2])) / np.sum(np.abs(exp[2]))
        }


def create_dataloader(base_directory='./data/train/',
                      num_workers=6,
                      batch_size=10,
                      sample_type=DefaultSample):
    dataset = Dataset(base_directory, sample_type=sample_type)

    return torch.utils.data.DataLoader(dataset,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       num_workers=num_workers)


def create_identity_dataloader(num_samples=30,
                               num_workers=6,
                               batch_size=10,
                               shape=(3, 64, 64),
                               noise_std=10,
                               max_val=100):
    dataset = IdentityDataset(num_samples=num_samples,
                              shape=shape,
                              noise_std=noise_std,
                              max_val=max_val)

    return torch.utils.data.DataLoader(dataset,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       num_workers=num_workers)


def initialize_network(loss_fn,
                       data_gen,
                       network_type=unet.SimpleNet,
                       network_kwargs={},
                       num_initializations=10,
                       device='cpu',
                       zero_geom=True):
    best_loss = np.inf
    best_net = None

    for _ in range(num_initializations):
        network = network_type(**network_kwargs)

        network.to(device)
        network.float()
        # network.float()

        if num_initializations == 1:
            return network

        loss = compute_average_loss(network,
                                    loss_fn,
                                    data_gen,
                                    device=device,
                                    zero_geom=zero_geom)
        if loss < best_loss:
            best_net = network

    return best_net


def create_checkpointers(training_problem: TrainingProblem,
                         loss_registry: LossRegistry,
                         save_every=50,
                         print_every=5,
                         checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                         end_character='\r',
                         timer=None,
                         zero_geom=True):

    manager = ModelManager(checkpoint_dir)
    state_checkpointer = StateCheckpointer(checkpoint_dir, save_every,
                                           training_problem, manager,
                                           loss_registry)

    plot_checkpointer = PlotCheckpointer(checkpoint_dir, save_every,
                                         loss_registry)

    epoch_number_checkpointer = EpochNumberCheckpointer(
        checkpoint_dir, loss_registry)

    loss_checkpointer = LossCheckpointer(checkpoint_dir,
                                         1,
                                         loss_registry,
                                         training_problem,
                                         zero_geom=zero_geom)

    checkpointers = [
        epoch_number_checkpointer,
        loss_checkpointer,
        state_checkpointer,
        plot_checkpointer,
    ]

    if print_every:
        if timer is None:
            timer = Timer()
            timer.start()
        stdout_checkpointer = StdoutCheckpointer(checkpoint_dir, print_every,
                                                 loss_registry, timer,
                                                 end_character)
        checkpointers.append(stdout_checkpointer)

    return CombineCheckpointer(checkpoint_dir, checkpointers)


def create_identity_training_problem(num_initializations=10,
                                     network_type=unet.SimpleNet):
    loss_fn = F.mse_loss

    training_data = create_identity_dataloader(num_samples=30)
    validation_data = create_identity_dataloader(num_samples=10)

    network = initialize_network(loss_fn,
                                 training_data,
                                 network_type=network_type,
                                 num_initializations=num_initializations)

    optimizer = torch.optim.Adam(network.parameters())

    training_problem = TrainingProblem(network, optimizer, loss_fn,
                                       training_data, validation_data)

    return training_problem


def create_identity_trainer(num_initializations=10,
                            network_type=unet.SimpleNet,
                            save_every=50,
                            print_every=5,
                            checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                            end_character='\r',
                            timer=None):
    loss_registry = LossRegistry()

    training_problem = create_identity_training_problem(
        num_initializations, network_type)

    checkpointer = create_checkpointers(training_problem, loss_registry,
                                        save_every, print_every,
                                        checkpoint_dir, end_character, timer)

    return Trainer(training_problem, checkpointer)


def load_identity_trainer(checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                          network_type=unet.SimpleNet):
    training_problem = create_identity_training_problem(
        num_initializations=1, network_type=network_type)
    state_checkpointer = StateCheckpointer(checkpoint_dir, training_problem,
                                           model_manager, loss_registry)


def per_channel_loss(inp, target, channel_weights=(1, 5, 10)):
    loss = torch.tensor(0., device='cuda')

    loss += F.l1_loss(inp[:, 0, :, :], target[:, 0, :, :]) * channel_weights[0]
    loss += F.l1_loss(inp[:, 1, :, :], target[:, 1, :, :]) * channel_weights[1]
    loss += F.l1_loss(inp[:, 2, :, :], target[:, 2, :, :]) * channel_weights[2]

    return loss


def relative_error_loss_norm(inp, target):
    mean_rel_error = torch.mean((inp - target)**2 / (inp**2 + target**2 + 1))
    loss = torch.atan(mean_rel_error)
    loss += np.pi / 2
    loss *= 10 / np.pi
    return loss


def create_training_problem(
        *,
        num_initializations=10,
        sample_type='default',
        data_dir='./data/',
        device='cpu',
        network_type=unet.UNet,
        batch_size=64,
        optimizer=torch.optim.Adam,
        optimizer_params={},
        num_workers=6,
        extend_network=None,
        loss_fn=per_channel_loss,
):
    data_dir = Path(data_dir)

    if extend_network:
        network = unet.UNetCD(unet=extend_network)
        network.float()
        network.to(device)
        sample_type = CDSample.freeze_scaling_factors(data_dir / 'train')
    else:
        network_kwargs = {'input_channels': 2, 'out_channels': 3}
        if sample_type == 'default':
            network_kwargs = {'input_channels': 2, 'out_channels': 4}
            sample_type = DefaultSample
        elif sample_type == 'no_cd':
            sample_type = VelPressSample
        elif sample_type == 'scaled':
            sample_type = NormalizedVelPressSample.freeze_scaling_factors(
                data_dir / 'train')
        elif sample_type == 'force':
            sample_type = CDSample.freeze_scaling_factors(data_dir / 'train')

    training_data = create_dataloader(base_directory=data_dir / 'train',
                                      sample_type=sample_type,
                                      batch_size=batch_size,
                                      num_workers=num_workers)
    validation_data = create_dataloader(base_directory=data_dir / 'val',
                                        sample_type=sample_type,
                                        batch_size=batch_size,
                                        num_workers=num_workers)
    test_data = None
    test_dir = data_dir / 'test'
    if test_dir.exists():
        test_data = create_dataloader(base_directory=test_dir,
                                      sample_type=sample_type,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

    if not extend_network:
        network = initialize_network(loss_fn,
                                     training_data,
                                     network_type=network_type,
                                     network_kwargs=network_kwargs,
                                     num_initializations=num_initializations,
                                     device=device,
                                     zero_geom=True)

    optimizer = optimizer(network.parameters(), **optimizer_params)

    training_problem = TrainingProblem(network,
                                       optimizer,
                                       loss_fn,
                                       training_data,
                                       validation_data,
                                       test_data=test_data,
                                       device=device)

    return training_problem


def create_trainer(data_dir='./data',
                   batch_size=64,
                   num_initializations=10,
                   save_every=50,
                   print_every=5,
                   checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                   end_character='\r',
                   timer=None,
                   device='cpu',
                   sample_type='scaled',
                   num_workers=6,
                   optimizer=torch.optim.Adam,
                   optimizer_params={'lr': 1e-3},
                   network_type=unet.UNet,
                   loss_fn=per_channel_loss,
                   extend_network=False):
    loss_registry = LossRegistry()

    training_problem = create_training_problem(
        num_initializations=num_initializations,
        device=device,
        loss_fn=loss_fn,
        data_dir=data_dir,
        network_type=network_type,
        sample_type=sample_type,
        batch_size=batch_size,
        num_workers=num_workers,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        extend_network=extend_network)

    checkpointer = create_checkpointers(training_problem,
                                        loss_registry,
                                        save_every,
                                        print_every,
                                        checkpoint_dir,
                                        end_character,
                                        timer,
                                        ezero_geom=bool(extend_network))

    return Trainer(training_problem, checkpointer)


def load_trainer(checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
                 batch_size=128,
                 sample_type='scaled',
                 data_dir='./data/',
                 save_every=50,
                 print_every=5,
                 end_character='\r',
                 num_workers=6,
                 timer=None,
                 device='cpu',
                 network_type=unet.UNet,
                 optimizer=torch.optim.Adam,
                 optimizer_params={},
                 loss_fn=per_channel_loss,
                 checkpoint_file=None):
    model_manager = ModelManager(checkpoint_dir)
    loss_registry = LossRegistry()

    training_problem = create_training_problem(
        num_initializations=1,
        loss_fn=loss_fn,
        data_dir=data_dir,
        batch_size=batch_size,
        network_type=network_type,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        num_workers=num_workers,
        sample_type=sample_type,
        device=device)

    state_checkpointer = StateCheckpointer(checkpoint_dir, None,
                                           training_problem, model_manager,
                                           loss_registry)

    if checkpoint_file is None:
        state_checkpointer.load_last_model()
    else:
        state_checkpointer.load_model(checkpoint_file)

    checkpointer = create_checkpointers(training_problem,
                                        loss_registry,
                                        save_every,
                                        print_every,
                                        checkpoint_dir,
                                        end_character,
                                        timer,
                                        zero_geom=True)

    return Trainer(training_problem, checkpointer)


if __name__ == '__main__':
    tr = train.load_trainer(  #num_initializations=1,
        batch_size=128,
        device='cuda',
        print_every=1,
        save_every=10,
        end_character='\n')

    tr = train.create_trainer(num_initializations=50,
                              batch_size=128,
                              device='cuda',
                              print_every=1,
                              num_workers=128,
                              save_every=10,
                              end_character='\n')

    for param_group in tr.problem.optimizer.param_groups:
        param_group['lr'] = 1e-6  # last

    tr = train.create_trainer(num_initializations=30,
                              network_type=train.SimpleNet,
                              batch_size=2048,
                              device='cuda',
                              print_every=1,
                              save_every=5,
                              num_workers=6,
                              end_character='\n')

    tr = train.load_trainer(batch_size=128,
                            device='cuda',
                            print_every=2,
                            save_every=2,
                            sample_type=None,
                            end_character='\n')

    tr.epochs(10)
