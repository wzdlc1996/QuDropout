import problems
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from typing import Callable, Tuple


def strToBarray(x: str) -> np.ndarray:
    return np.array([int(z) for z in x])


def barrayToStr(x: np.ndarray) -> str:
    return "".join([str(z) for z in x])


def hammDist(x: np.ndarray, y: np.ndarray) -> int:
    return len(np.flatnonzero(x - y))


def hammNeighbor(x: np.ndarray) -> list:
    res = []
    for i, v in enumerate(x):
        z = np.copy(x)
        z[i] = 1 - v
        res.append(z)
    return res


def randFlip(_x: np.ndarray) -> np.ndarray:
    x = np.copy(_x)
    flipat = np.random.randint(len(x))
    x[flipat] = 1 - x[flipat]
    return x


def SimAnnealing(prob: problems._spinGlassProblem, t_max: int, T: float) -> Tuple[np.ndarray, float, bool]:
    """
    Simulating annealing for the problem

    :param prob: should be the instance of problems._spinGlassProblem
    :param t_max: is the length for annealing process
    :param T: is the temperature factor
    :return: (configuration, energy, is_success)
    """
    size, spec = prob.info()
    state = np.random.randint(0, 2, size)
    # state = strToBarray(list(prob.refcfg.keys())[np.argmax(spec)])
    t = 0
    while t < t_max:
        temper = (1 - t / t_max) * T
        # nex = hammNeighbor(state)
        # nex = nex[np.random.randint(len(nex))]
        nex = randFlip(state)
        enex = prob(nex)
        enow = prob(state)
        if enex < enow:
            state = nex
        elif np.exp(- (enex - enow) / temper) > np.random.rand():
            state = nex
        t += 1
    return state, prob(state), prob(state) == 0


def SimAnnealForSpec(spec: np.ndarray, nsize: int, t_max: int, T: float) -> Tuple[np.ndarray, float, bool]:
    """
    Run simulating annealing for a single spectrum

    :param spec: spectrum
    :param nsize: system size
    :param t_max: maximum
    :param T: temperature factor
    :return: (configuration, energy, is_success)
    """
    prob = problems._spinGlassProblem(nsize)
    prob.spec = spec
    return SimAnnealing(prob, t_max, T)


class SimAnnealer:
    def __init__(
            self,
            spec: np.ndarray,
            nsize: int,
            checkSol: Callable[[np.ndarray], bool],
            t_max: int = 500,
            T: float = 0.1
    ):
        self.spec = spec
        self.nsize = nsize
        self.t_max = t_max
        self.T = T
        self.checkSol = checkSol

    def __call__(self, x) -> bool:
        conf, _, _ = SimAnnealForSpec(self.spec, self.nsize, self.t_max, self.T)
        return self.checkSol(conf)


def specPreview(prob: problems._spinGlassProblem, titl="TBD") -> None:
    sols = prob.solution()
    confInfo = {}
    def distToSol(cfg):
        return min([hammDist(strToBarray(cfg), strToBarray(x)) for x in sols])
    for cfg in prob.refcfg:
        dist = distToSol(cfg)
        if dist not in confInfo:
            confInfo.update({dist: [prob(cfg)]})
        else:
            confInfo[dist].append(prob(cfg))
    dataX = list(confInfo.keys())
    dataY = [np.average(x) for x in list(confInfo.values())]
    errYmin = [np.average(x) - min(x) for x in list(confInfo.values())]
    errYmax = [max(x) - np.average(x) for x in list(confInfo.values())]
    plt.errorbar(dataX, dataY, yerr=[errYmin, errYmax], fmt="o")
    plt.title(titl)
    plt.xlabel("Distance to Solution")
    plt.ylabel("Mean Energy")
    plt.show()


def SimAnnealStat(spec: np.ndarray, sampN: int, t_max: int, T: float) -> Tuple[int, int]:
    """
    Simulating annealing statistics for spectrum

    :param spec: spectrum
    :param sampN: sample size
    :param t_max: simulating annealing length
    :param T: temperature
    :return: (success_number, total_number)
    """
    dim = len(spec)
    nsize = int(np.log2(dim))

    sols = [
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.ones(16) - np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]

    def checkSol(conf: np.ndarray) -> bool:
        return (conf == sols[0]).all() or (conf == sols[1]).all()


    annler = SimAnnealer(spec, nsize, checkSol, t_max, T)
    pool = mp.Pool(processes=8)
    z = pool.map(annler, [0] * sampN)
    pool.close()
    pool.join()
    return len([x for x in z if x]), len(z)


if __name__ == '__main__':
    prb = problems.ex_test_hard
    print(prb)
    n_exp = 200
    n_sa = 400
    print("--------No cutoff--------")
    succ = 0
    for i in range(n_exp):
        _, e, _ = SimAnnealing(prb, n_sa, 10000)
        if e == min(prb.spec):
            succ += 1
    print(f"Success probability: {succ / n_exp}")
    specPreview(prb, "No Cutoff")
    # print("--------With cutoff--------")
    # prb.cutoff(100)
    # specPreview(prb, "With Cutoff")
    # succ = 0
    # for i in range(n_exp):
    #     _, e = SimAnnealing(prb, n_sa)
    #     if e == min(prb.spec):
    #         succ += 1
    # print(f"Success probability: {succ / n_exp}")




