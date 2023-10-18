import math
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import control  # pip install python-control, pip install slycot (optional)
from lti import drss_matrices, dlsim


class LinearDynamicalDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=500, strictly_proper=True, dtype="float32", normalize=True):
        super(LinearDynamicalDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize

    def __iter__(self):
        while True:  # infinite dataset
            # for _ in range(1000):
            sys = control.drss(states=self.nx,
                               inputs=self.nu,
                               outputs=self.ny,
                               strictly_proper=self.strictly_proper)
            u = np.random.randn(self.nu, self.seq_len).astype(self.dtype)  # C, T as python-control wants
            y = control.forced_response(sys, T=None, U=u, X0=0.0)
            u = u.transpose()  # T, C
            y = y.y.transpose().astype(self.dtype)  # T, C
            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            yield torch.tensor(y), torch.tensor(u)


class WHDataset(IterableDataset):
    def __init__(self, nx=5, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(WHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.system_seed = system_seed
        self.data_seed = data_seed
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.mdlargs = mdlargs

    def __iter__(self):

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        n_in = 1
        n_out = 1
        n_hidden = 32
        n_skip = 200

        if self.fixed_system:  # same model at each step, generate only once!
            w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=self.strictly_proper,
                               rng=self.system_rng,
                               **self.mdlargs)

            G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=False,
                               rng=self.system_rng,
                               **self.mdlargs)

        while True:  # infinite dataset

            if not self.fixed_system:  # different model for different instances!
                w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
                b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
                w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
                b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

                G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=self.strictly_proper,
                                   rng=self.system_rng,
                                   **self.mdlargs)

                G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=False,
                                   rng=self.system_rng,
                                   **self.mdlargs)

            #u = np.random.randn(self.seq_len + n_skip, 1)  # input to be improved (filtered noise, multisine, etc)
            u = self.data_rng.normal(size=(self.seq_len + n_skip, 1))

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            yield torch.tensor(y), torch.tensor(u)


class PWHDataset(IterableDataset):
    def __init__(self, nx=50, nu=1, ny=1, nbr=10, seq_len=1024, n_hidden=256, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(PWHDataset).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nbr = nbr
        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order
        self.mdlargs = mdlargs
        self.system_seed = system_seed
        self.data_seed = data_seed
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation

    def sample_system(self):

        w1 = self.system_rng.normal(size=(self.n_hidden, self.nbr)) / np.sqrt(self.nbr) * 1.0
        b1 = self.system_rng.normal(size=(1, self.n_hidden)) * 1.0
        w2 = self.system_rng.normal(size=(self.nbr, self.n_hidden)) / np.sqrt(
            self.n_hidden) * 5 / 3  # compensates previous tanh
        b2 = self.system_rng.normal(size=(1, self.nbr)) * 1.0

        G1 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                           inputs=self.nu,
                           outputs=self.nbr,
                           strictly_proper=self.strictly_proper,
                           rng=self.system_rng,
                           **self.mdlargs)

        G2 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                           inputs=self.nbr,
                           outputs=self.ny,
                           strictly_proper=False,  # no delay here (if one is desired, put it in G1)
                           rng=self.system_rng,
                           **self.mdlargs)

        return w1, b1, w2, b2, G1, G2

    def __iter__(self):

        w1, b1, w2, b2, G1, G2 = self.sample_system()

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        n_skip = 200
        while True:

            u = self.data_rng.normal(size=(self.seq_len + n_skip, self.nu))

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            if not self.fixed_system:  # change system!
                w1, b1, w2, b2, G1, G2 = self.sample_system()

            yield torch.tensor(y), torch.tensor(u)



def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.data_rng = np.random.default_rng(dataset.data_seed + 1000*worker_id)
    dataset.system_rng = np.random.default_rng(dataset.system_seed + 1000*worker_id)
    #print(worker_id, worker_info.id)


class NonInfinitePWHDataset(IterableDataset):
    def __init__(self, seq_num=100, nx=50, nu=1, ny=1, nbr=10, seq_len=1024, n_hidden=256, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(PWHDataset).__init__()
        self.seq_num = seq_num
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nbr = nbr
        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order
        self.mdlargs = mdlargs
        self.system_seed = system_seed
        self.data_seed = data_seed
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation

    def sample_system(self):

        w1 = self.system_rng.normal(size=(self.n_hidden, self.nbr)) / np.sqrt(self.nbr) * 1.0
        b1 = self.system_rng.normal(size=(1, self.n_hidden)) * 1.0
        w2 = self.system_rng.normal(size=(self.nbr, self.n_hidden)) / np.sqrt(
            self.n_hidden) * 5 / 3  # compensates previous tanh
        b2 = self.system_rng.normal(size=(1, self.nbr)) * 1.0

        G1 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                           inputs=self.nu,
                           outputs=self.nbr,
                           strictly_proper=self.strictly_proper,
                           rng=self.system_rng,
                           **self.mdlargs)

        G2 = drss_matrices(states=self.system_rng.integers(1, self.nx + 1) if self.random_order else self.nx,
                           inputs=self.nbr,
                           outputs=self.ny,
                           strictly_proper=False,  # no delay here (if one is desired, put it in G1)
                           rng=self.system_rng,
                           **self.mdlargs)

        return w1, b1, w2, b2, G1, G2

    def __iter__(self):

        w1, b1, w2, b2, G1, G2 = self.sample_system()

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        tensor_list_u = []
        tensor_list_y = []
        n_skip = 200
        for _ in range(self.seq_num):
            u = self.data_rng.normal(size=(self.seq_len + n_skip, self.nu))

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            if not self.fixed_system:  # change system!
                w1, b1, w2, b2, G1, G2 = self.sample_system()

            tensor_list_y.append(torch.tensor(y))
            tensor_list_u.append(torch.tensor(u))

        sequences_y = torch.stack(tensor_list_y)
        sequences_u = torch.stack(tensor_list_u)

        i = -1
        while True:  # infinite repetition of the fixed sequences
            i += 1
            i = i % self.seq_num
            yield sequences_y[i], sequences_u[i]


class NonInfiniteWHDataset(IterableDataset):
    def __init__(self, seq_num=100, nx=5, nu=1, ny=1, seq_len=600, random_order=True,
                 strictly_proper=True, normalize=True, dtype="float32",
                 fixed_system=False, system_seed=None, data_seed=None, **mdlargs):
        super(WHDataset).__init__()
        self.seq_num = seq_num
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.seq_len = seq_len
        self.strictly_proper = strictly_proper
        self.dtype = dtype
        self.normalize = normalize
        self.strictly_proper = strictly_proper
        self.random_order = random_order  # random number of states from 1 to nx
        self.system_rng = np.random.default_rng(system_seed)  # source of randomness for model generation
        self.data_rng = np.random.default_rng(data_seed)  # source of randomness for model generation
        self.fixed_system = fixed_system  # same model at each iteration (classical identification)
        self.mdlargs = mdlargs

    def __iter__(self):

        # A simple ff neural network
        def nn_fun(x):
            out = x @ w1.transpose() + b1
            out = np.tanh(out)
            out = out @ w2.transpose() + b2
            return out

        n_in = 1
        n_out = 1
        n_hidden = 32
        n_skip = 200

        if self.fixed_system:  # same model for all data sequences!
            w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
            b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
            w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
            b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

            G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=self.strictly_proper,
                               rng=self.system_rng,
                               **self.mdlargs)

            G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                               inputs=1,
                               outputs=1,
                               strictly_proper=False,
                               rng=self.system_rng,
                               **self.mdlargs)

        tensor_list_u = []
        tensor_list_y = []
        for i in range(self.seq_num):   # generate a fixed amount of sequences

            if not self.fixed_system:  # different model for different instances!
                w1 = self.system_rng.normal(size=(n_hidden, n_in)) / np.sqrt(n_in) * 5 / 3
                b1 = self.system_rng.normal(size=(1, n_hidden)) * 1.0
                w2 = self.system_rng.normal(size=(n_out, n_hidden)) / np.sqrt(n_hidden)
                b2 = self.system_rng.normal(size=(1, n_out)) * 1.0

                G1 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=self.strictly_proper,
                                   rng=self.system_rng,
                                   **self.mdlargs)

                G2 = drss_matrices(states=self.system_rng.integers(1, self.nx+1) if self.random_order else self.nx,
                                   inputs=1,
                                   outputs=1,
                                   strictly_proper=False,
                                   rng=self.system_rng,
                                   **self.mdlargs)

            # u = np.random.randn(self.seq_len + n_skip, 1)  # input to be improved (filtered noise, multisine, etc)
            u = self.data_rng.normal(size=(self.seq_len + n_skip, 1))

            # G1
            y1 = dlsim(*G1, u)
            y1 = (y1 - y1[n_skip:].mean(axis=0)) / (y1[n_skip:].std(axis=0) + 1e-6)

            # F
            y2 = nn_fun(y1)

            # G2
            y3 = dlsim(*G2, y2)

            u = u[n_skip:]
            y = y3[n_skip:]

            if self.normalize:
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-6)

            u = u.astype(self.dtype)
            y = y.astype(self.dtype)

            tensor_list_y.append(torch.tensor(y))
            tensor_list_u.append(torch.tensor(u))

        sequences_y = torch.stack(tensor_list_y)
        sequences_u = torch.stack(tensor_list_u)

        i = -1
        while True:  # infinite repetition of the fixed sequences
            i += 1
            i = i % self.seq_num
            yield sequences_y[i], sequences_u[i]

            
if __name__ == "__main__":
    train_ds = WHDataset(nx=2, seq_len=4, mag_range=(0.5, 0.96),
                         phase_range=(0, math.pi / 3),
                         system_seed=42, data_seed=445, fixed_system=False)
    # train_ds = LinearDynamicalDataset(nx=5, nu=2, ny=3, seq_len=1000)
    train_dl = DataLoader(train_ds, batch_size=2, num_workers=10, worker_init_fn=seed_worker)
    batch_y, batch_u = next(iter(train_dl))
    batch_y, batch_u = next(iter(train_dl))
    print(batch_u.shape, batch_u.shape)