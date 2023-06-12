import satd_utils
import problems
import numpy as np
import random
import time
import torch

from satDropout_New import PrefDropoutBase

# easiest 10: [41, 196, 13, 39, 178, 82, 68, 84, 52, 113]
# succ_p: [0.735, 0.738, 0.744, 0.756, 0.768, 0.808, 0.811, 0.814, 0.874, 0.951]
# hardest 10: [25, 165, 16, 53, 118, 152, 161, 88, 130, 147]
# succ_p: [0.069, 0.077, 0.08, 0.095, 0.099, 0.103, 0.11, 0.116, 0.117, 0.121]
problemGetter = problems.GSATN16C112Getter
easyGetter = problems.GSATN16C101EasyGetter
inst = problemGetter[25]
clause_hard = inst["clause"]
clause_easy = easyGetter[0]["clause"]

modec = clause_easy
lossc = clause_hard


superParams = {
    "nsize": 16,
    "dropout_r": 0.5,
    "dep": 30,
    "subEpoch": 10,
    "maxEpoch": 200,
    "lr": 1e-2,
    "root_folder": "./DataCollec/n=16/CrossLoss",
    "randr": (-1., 1.),
    "comm": f"hard25 as model, easy00 as loss\n"
             "optim.LBFGS optimizer",
    "dir": "easy00-hard25-doub"
}


if __name__ == "__main__":
    # seed = time.time_ns()
    # random.seed(seed)
    # print(f"RandomSeed is: {seed}")
    # with open(f"{superParams['root_folder']}/{superParams['dir']}/seed.txt", "w") as f:
    #     f.write(f"RandomSeed for Init model is: {seed}")


    env = satd_utils.TracingEnv()

    nsize = superParams["nsize"]
    env.set_params(
        nsize=superParams["nsize"],
        dep=superParams["dep"],
        maxEpoch=superParams["maxEpoch"],
        # Is the time of dropout, also the time of changing criterion/model
        # i.e., (maxEpoch / dropt) is the epochs of subtrain: with fixed model
        # and loss function.
        dropt=superParams["subEpoch"],
        cls=lossc,
        comm=superParams["comm"]
    )

    env.set_device("cuda:0")
    env.set_mod(PrefDropoutBase)
    env.set_lr(superParams["lr"])

    def asOptim(param):
        return torch.optim.LBFGS(param, lr=env.lr)


    def prescript(module: PrefDropoutBase) -> PrefDropoutBase:
        module.setOptimizer(asOptim)
        probT = problems.HardGeneralSAT_normalized
        model_problem = probT(nsize, modec)
        loss_problem = probT(nsize, lossc)

        model_spec = model_problem.info()[1]
        loss_spec = loss_problem.info()[1]

        module.setLayerSpec([model_spec] * module.depth)
        module.setLoss(loss_spec)
        return module


    env.set_prescript(prescript)
    # env.rand_on(superParams['randr'])
    env.rand_off()

    env.make_comment()
    with satd_utils.ch_dir(superParams["root_folder"]):
        import os
        direc = env.run(lite=True)
        np.save(f"./{direc}/modl_clauses", modec)
        np.save(f"./{direc}/loss_clauses", lossc)
        os.replace(direc, superParams["dir"])


