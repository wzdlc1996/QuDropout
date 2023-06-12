"""
Is the statistics for sampling of the final algorithm.
"""

import satd_utils
import problems
import numpy as np
import random
import torch
import sys
import os
import time

from satDropout_New import PrefDropoutBase
from io import StringIO

problemGetter = problems.GSATN16C112Getter
inst = problemGetter[25]
clause, low_conf, succ_p = inst["clause"], inst["conf"], inst["succ_p"]

superParams = {
    "nsize": 16,
    "dropout_r": 0.5,  # 1.0 means preserve all clauses
    "deps-sampl": {
        # 5: 100,
        # 10: 100,
        # 15: 100,
        # 20: 100,
        # 25: 50,
        # 30: 50,
        # 35: 50,
        # 40: 50,
        # 45: 30,
        # 50: 30
        5: 50,
        10: 50,
        15: 50,
        20: 50,
        25: 30,
        30: 30,
        35: 30,
        40: 30,
        45: 15,  # next start
        50: 15
    },
    "subEpoch": 10,
    "maxEpoch": 200,
    "lr": 1e-2,
    # "cls": problems.GSATN16C128HardestInstance,
    "cls": clause,
    "root_folder": "./DataCollec/n=16/intSampling",
    "randr": (-np.pi, np.pi),
    "batch_size": 16,
    "gpu_avail": [0, 1, 2, 3, 4, 5]
}


def genDropoutProblem(nes, ess, size, probT, p: float):
    diff = 0
    while diff == 0:
        newcls = random.sample(nes, int(p * len(nes)))
        newcls += ess
        prb = probT(size, newcls)
        size, spec = prb.info()
        diff = max(spec) - min(spec)
    # return no_normalized, normalized spectrum
    return prb.spec_origin, prb.info()[1]


def specGen(samplLen: int):
    probT = problems.HardGeneralSAT_normalized
    ratio = superParams["dropout_r"]
    nsize = superParams["nsize"]
    model_problem = probT(nsize, superParams["cls"])

    # Randomly use half of the low lying configurations
    usdconf = low_conf
    # random.shuffle(low_conf)
    # usdconf = low_conf[:int(len(low_conf) / 2)]

    clss = superParams["cls"]
    ess = []
    nes = []
    ess_cls = model_problem.essential_clauses_by_given_confs(usdconf)
    for cls in clss:
        if cls in ess_cls:
            ess.append(cls)
        else:
            nes.append(cls)

    sub_specs_tup = [
        genDropoutProblem(nes, ess, nsize, probT, ratio)
        for _ in range(samplLen)
    ]

    sub_specs = [sub[1] for sub in sub_specs_tup]
    sub_specs_ori = [sub[0] for sub in sub_specs_tup]
    return sub_specs, sub_specs_ori


def makeEnv(gpuid: int=1, dep: int=1):
    env = satd_utils.TracingEnv()
    env.set_params(
        nsize=superParams["nsize"],
        dep=dep,
        maxEpoch=superParams["maxEpoch"],
        dropt=superParams["subEpoch"],
        cls=superParams["cls"],
        comm="optim.LBFGS optimizer.\n"
    )
    env.set_device(f"cuda:{gpuid}")
    env.set_mod(PrefDropoutBase)
    env.set_lr(superParams["lr"])
    env.rand_on(superParams["randr"])
    return env


def asOptim(param):
    return torch.optim.LBFGS(param, lr=superParams["lr"])


def _genseed() -> int:
    # use 4-th Bell Prime
    return (time.time_ns() * os.getpid()) % 27644437


def _sampleParFolderPatter(subdir, dep):
    return f"{superParams['root_folder']}/{subdir}/p={dep}"


def _envRunner(**kwargs):
    env = kwargs["env"]
    samp_id = kwargs["samp_id"]
    dep = kwargs["dep"]
    seed = kwargs["seed"]
    subdir = kwargs["subdir"]
    if "subspecs" in kwargs:
        subspecs = kwargs["subspecs"]
    else:
        subspecs = None

    env.make_comment()
    with satd_utils.ch_dir(_sampleParFolderPatter(subdir, dep)):
        temp = sys.stdout
        sys.stdout = oneout = StringIO()
        fold = env.run(lite=True)
        sys.stdout = temp

        res = oneout.getvalue()
        with open(f"{fold}/console.txt", "w") as f:
            f.write(res)

        with open(f"{fold}/rnd_seed.txt", "w") as f:
            f.write(f"RandomSeed for Init model is: {seed}")

        if subspecs is not None:
            np.save(f"{fold}/usd_subspecs", subspecs)

        os.replace(fold, "sample_%.3d" % samp_id)


def singleRun_RegQAOA(gpuid: int, samp_id: int, dep: int):
    seed = _genseed()
    random.seed(seed)
    env = makeEnv(gpuid, dep)

    def setoptim(module: PrefDropoutBase):
        module.setOptimizer(asOptim)
        return module

    env.set_prescript(setoptim)
    _envRunner(
        env=env,
        samp_id=samp_id,
        dep=dep,
        seed=seed,
        subdir="RegQAOA"
    )


def singleRun_DropoutRND(gpuid: int, samp_id:int, dep: int):
    seed = _genseed()
    random.seed(seed)
    env = makeEnv(gpuid, dep)

    subspecs, _ = specGen(samplLen=dep)

    def preset(module: PrefDropoutBase) -> PrefDropoutBase:
        module.setOptimizer(asOptim)
        module.setLayerSpec(subspecs)
        return module

    env.set_prescript(preset)
    _envRunner(
        env=env,
        samp_id=samp_id,
        dep=dep,
        seed=seed,
        subdir="DropRND",
        subspecs=subspecs
    )


def singleRun_DropoutHOMO(gpuid: int, samp_id: int, dep: int):
    seed = _genseed()
    random.seed(seed)
    env = makeEnv(gpuid, dep)

    subspecs, _ = specGen(samplLen=1)

    def preset(module: PrefDropoutBase) -> PrefDropoutBase:
        module.setOptimizer(asOptim)
        module.setLayerSpec([subspecs[0] for _ in range(dep)])
        return module

    env.set_prescript(preset)
    _envRunner(
        env=env,
        samp_id=samp_id,
        dep=dep,
        seed=seed,
        subdir="DropHOMO",
        subspecs=subspecs
    )


if __name__ == "__main__":
    seed = time.time_ns()
    random.seed(seed)
    print(f"RandomSeed is: {seed}")
    with open(f"{superParams['root_folder']}/seed.txt", "a") as f:
        f.write(f"RandomSeed for Init model is: {seed}\tTimestamp={time.ctime()}\n")

    #  There are six delicate gpus on the server
    avail_gpus = superParams["gpu_avail"]
    batch_size = superParams["batch_size"]

    import torch.multiprocessing as mp
    import mkl
    mkl.set_num_threads(1)
    mp.set_start_method("spawn")

    for dep in superParams["deps-sampl"]:
        for sub in ["DropHOMO", "DropRND", "RegQAOA"]:
            with satd_utils.ch_dir(_sampleParFolderPatter(sub, dep)):
                pass

    for dep, sampl in superParams["deps-sampl"].items():
        params = [
            (avail_gpus[i % len(avail_gpus)], i, dep)
            for i in range(sampl)
        ]

        params_list = [
            params[i:min(i+batch_size, len(params))]
            for i in range(0, len(params), batch_size)
        ]

        for mod in [singleRun_RegQAOA, singleRun_DropoutRND, singleRun_DropoutHOMO]:
            for pp in params_list:
                pool = mp.Pool(processes=batch_size)
                pool.starmap(mod, pp)
                pool.close()
                pool.join()



"""
1646804420548740197
[[0, 10, 8], [1, 12, 9], [1, 10, 8], [1, 4, 9]]
"""

"""
1646804477609117256
[[6, 1, 10], [6, 10, 8], [1, 11, 15], [2, 6, 9]]
"""