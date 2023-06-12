import numpy as np
import torch
import math
import torch.nn.functional as F


class _singleQubitGates(torch.nn.Module):
    def __init__(self, _size, _site=None):
        super().__init__()
        self.size = _size
        self.dim = 2 ** _size
        if _site is None:
            self.site = np.arange(self.size)
        elif isinstance(_site, list):
            self.site = _site
        elif isinstance(_site, int):
            self.site = [_site]
        else:
            raise ValueError("Invalid value for 'site'")


class GateX(_singleQubitGates):
    def __init__(self, _size, _site=None):
        super().__init__(_size, _site)
        ind = torch.arange(self.dim, dtype=torch.long)
        for x in self.site:
            bin_ind = np.repeat(np.tile([0, 1], 2 ** x), 2 ** (self.size - 1 - x))
            negb = np.arange(self.dim) + (1 - 2 * bin_ind) * 2 ** (self.size - 1 - x)
            ind = ind[negb]
        self.register_buffer("ind", ind)

    def forward(self, x):
        return x[:, self.ind]


class GateZ(_singleQubitGates):
    def __init__(self, _size, _site=None):
        super().__init__(_size, _site)
        sign = np.ones(self.dim)
        for x in self.site:
            sign *= np.repeat(np.tile([1, -1], 2 ** x), 2 ** (self.size - 1 - x))
        self.register_buffer("amp", torch.from_numpy(sign))
        # self.amp = torch.from_numpy(sign)

    def forward(self, x):
        return self.amp * x


class GateHadamard(_singleQubitGates):
    # Need Qubit-wise Parallelization
    def __init__(self, _size, _site=None):
        super().__init__(_size, _site)
        N = self.size
        bin_site_inds = [np.repeat(np.tile([0, 1], 2 ** site), 2 ** (N - 1 - site)) for site in range(N)]
        sign_inds = np.array([np.repeat(np.tile([1, -1], 2 ** site), 2 ** (N - 1 - site)) for site in range(N)])
        indx = np.array([np.arange(self.dim, dtype=int) + (1 - 2 * bin_site_inds[i]) * 2 ** (N - 1 - i) for i in range(N)])
        self.register_buffer("signs", torch.tensor(sign_inds))
        self.register_buffer("indxs", torch.tensor(indx))
        # self.signs = torch.tensor(sign_inds)
        # self.indxs = torch.tensor(indx)

    def forward(self, x):
        for i in self.site:
            x = 1. / math.sqrt(2) * (self.signs[i] * x + x[:, self.indxs[i]])
        return x


class _doubleQubitGates(torch.nn.Module):
    def __init__(self, _size, _site=None):
        super().__init__()
        self.size = _size
        self.dim = 2 ** _size
        if _site is None:
            self.site = [(i, j) for i in range(_size) for j in range(_size) if i != j]
        elif isinstance(_site, list):
            self.site = _site
        elif isinstance(_site, tuple):
            self.site = [_site]
        else:
            raise ValueError("Invalid value for 'site' in double qubit gate")


class GateCNOT(_doubleQubitGates):
    # Need Qubit-wise Parallelization
    def __init__(self, _size, _site=None):
        super().__init__(_size, _site)
        usd_z = {}
        usd_x = {}
        for s, sp in self.site:
            if s not in usd_z:
                usd_z[s] = GateZ(self.size, s)
            if sp not in usd_x:
                usd_x[sp] = GateX(self.size, sp)

        self.usd_z = usd_z
        self.usd_x = usd_x

    def forward(self, x):
        for s, sp in self.site:
            zx = self.usd_z[s](x)
            x0 = (x + zx) / 2
            x1 = self.usd_x[sp](x - zx) / 2
            x = x0 + x1
        return x


class GateCNOTOptimized(_doubleQubitGates):
    """
    20 times faster than GateCNOT
    """
    def __init__(self, _size, _site=None):
        super().__init__(_size, _site)
        usd_s = {}
        N = self.size
        ind_n = torch.arange(self.dim, dtype=torch.long)
        for ss in self.site:
            for s in ss:
                if s not in usd_s:
                    usd_s[s] = torch.tensor(np.repeat(np.tile([0, 1], 2 ** s), 2 ** (N - 1 - s)))

            s, sp = ss
            ind_n += (- usd_s[sp] + torch.logical_xor(usd_s[sp], usd_s[s])) * 2 ** (N - 1 - sp)

        self.register_buffer("ind", ind_n)

    def forward(self, x):
        return x[:, self.ind]


class EvoIsingHamiltonian(torch.nn.Module):
    # 2 times slower than numpy implement
    def __init__(self, _size, _spec, _tini=1.):
        super().__init__()
        self.size = _size
        self.dim = 2 ** _size
        self.register_buffer("spec", torch.tensor(_spec, dtype=torch.float))
        self.t = torch.nn.Parameter(torch.tensor(_tini))
        # self.register_buffer("t", torch.nn.Parameter(torch.tensor(_tini)))

    def forward(self, x):
        sp = self.t * self.spec
        exp = torch.cos(sp) - 1.j * torch.sin(sp)
        return exp * x


def _enerCal(confstr, hamil):
    ener = 0
    conf = [int(x) for x in confstr]
    for i, x in enumerate(hamil["z"]):
        ener += (1 - 2 * conf[i]) * x
    for (i, j), k in hamil["zz"].items():
        ener += (1 - 2 * conf[i]) * (1 - 2 * conf[j]) * k
    return ener


def _hamil2spec(hamil) -> np.array:
    n = len(hamil["z"])
    return np.array([_enerCal(np.binary_repr(x, n), hamil) for x in range(2 ** n)])


class EvoXMixing(torch.nn.Module):
    # 2 times slower than numpy implement
    def __init__(self, _size, _tini=None):
        super().__init__()
        self.size = _size
        self.dim = 2 ** _size
        self.hadam = GateHadamard(self.size)
        homofield = {"z": np.ones(self.size), "zz": {}}
        spec = [_enerCal(np.binary_repr(x, self.size), homofield) for x in range(self.dim)]
        if _tini is not None:
            self.evolu = EvoIsingHamiltonian(self.size, spec, _tini)
        else:
            self.evolu = EvoIsingHamiltonian(self.size, spec)

    def forward(self, x):
        x = self.hadam(x)
        x = self.evolu(x)
        x = self.hadam(x)
        return x


class EvoXMixingNormalized(EvoXMixing):
    """
    Is the EvoXMixing layer with normalized spectrum.
    """
    def __init__(self, _size, _tini=None):
        super().__init__(_size, _tini)
        spec = self.evolu.spec.clone().detach()
        spec = spec - min(spec)
        spec /= max(spec) - min(spec)
        self.evolu.register_buffer("spec", spec)


class FixedXMixing(torch.nn.Module):
    # product(1 + i X) gate
    def __init__(self, _size):
        super().__init__()
        self.size = _size
        self.dim = 2 ** _size
        self.hadam = GateHadamard(self.size)
        self.gatex = GateX(self.size)

    def forward(self, x):
        for _ in range(self.size):
            x = (x + 1.j * self.gatex(x)) / math.sqrt(2)
        return x


class ExpectValue(torch.nn.Module):
    def __init__(self, _size, _spec):
        super().__init__()
        self.size = _size
        self.register_buffer("spec", torch.tensor(_spec, dtype=torch.float))

    def forward(self, x) -> torch.Tensor:
        lens = len(x)

        # torch.matmul is memory expensive
        # return torch.real(torch.trace(torch.matmul(torch.conj(x).T, self.spec * x))) / lens
        return torch.real(torch.sum(torch.conj(x) * (self.spec * x))) / lens


if __name__ == '__main__':
    # import time
    # nn = 3
    # hh = {"z": np.random.randn(nn), "zz": {(i, j): np.random.randn() for i in range(nn) for j in range(i, nn)}}
    # spec = [_enerCal(np.binary_repr(x, nn), hh) for x in range(2 ** nn)]
    # gx = ExpectValue(nn, spec)
    # inp = torch.randn((1, 2 ** nn)) * 1.j
    # # inp = torch.eye(2**nn)
    # ref = np.array(inp)
    #
    # str = time.time()
    # res1 = gx(inp)
    # end = time.time()
    # print(f"t: {end - str}; res: {res1 - min(spec)}")

    import time

    nn = 5
    x = torch.rand((1, 2**nn), dtype=torch.complex64)
    v = EvoXMixing(nn, 1.)
    w = EvoXMixingNormalized(nn, 1. * (2. * nn))
    st = time.time()
    print(v(x))
    ed = time.time()
    print(w(x) * np.exp(1.j * nn))
    t3 = time.time()
    print(f"origin: {ed - st}\tnormalized: {t3 - ed}")


    #
    # st = phy.qState(num=3, N=nn)
    # st._set(ref)
    # str = time.time()
    # for i, x in enumerate(hh["z"]):
    #     st.hadam(i)
    #     st.expZ(i, 3.)
    #     st.hadam(i)
    # # for (i, j), x in hh["zz"].items():
    # #     st.expZZ(i, j, x)
    # end = time.time()
    # print(f"t: {end - str}; res: {st.wf[:, :5]}")




