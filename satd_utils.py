import math
import numpy as np
import os
import json
import time
import problems
import torch
import satDropout_New as satd

from pathlib import Path


import mkl
mkl.set_num_threads(1)


def timenow():
    """
    Get the formatted time string to use as folder name
    :return:
    """
    return time.strftime("%b_%d_%H:%M:%S", time.gmtime())


def stdout_redir_dec(outfile):
    def dec(func):
        def wrapper(*args, **kwargs):
            import sys
            temp = sys.stdout
            with open(outfile, "w") as f:
                sys.stdout = f
                r = func(*args, **kwargs)
                sys.stdout = temp
            return r

        return wrapper
    return dec


def readLoss(fold):
    """
    Gather the epoch_eloss information in the fold
    :param fold: directory of the single_project
    :return res: numpy.ndarray, of eloss
    :return brkpt: list, of the dropout time position
    """
    res = []
    brkpt = []
    i = 0
    svd = os.listdir(fold)
    while True:
        toread = "epoch_eloss_%.3d.txt" % i
        if toread not in svd:
            break
        data = np.loadtxt(f"{fold}/{toread}")
        # help to fix logging before 09/14, since there are empty file in epoch_eloss series
        try:
            res.extend(data[:, 1])
        except:
            pass
        brkpt.append(len(res))
        i += 1
    return res, brkpt


class ch_dir(object):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.temp_now = os.getcwd()

    def __enter__(self):
        if not os.path.exists(self.path):
            path = Path(self.path)
            path.mkdir(parents=True)
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.temp_now)


class _envBase:
    def __init__(self, nsize=12, dep=20, problemTemplate=problems.HardGeneralSAT_normalized):
        self.nsize = nsize  # Is the size of instance
        self.dep = dep  # Is the depth of model
        self.problemTemplate = problemTemplate  # Is the problem template, stored in problems.py

        self.maxEpoch = 400  # Is the maximum epoch number
        self.dropt = 20  # Is the times of dropout, also the times of tracing

        self.p = 1.  # Is the default dropout p value
        self.sch = None  # Is the p-scheduler
        self.lr = 0.1  # Is the default learning rate for training
        self.randini = False  # Whether to randomly initialize the model
        self.randr = None

        self.dev = "cpu"  # Is the device name

        self.comm = ""  # Is the comment to store

        self.cls = None

        self.mod = satd.SATDropOut
        self.checkpoint = None

        self.pre_script = None

    def set_mod(self, mod):
        self.mod = mod
        return self

    def set_checkpoint(self, dir_name):
        self.checkpoint = os.path.abspath(dir_name)
        return self

    def set_prescript(self, func):
        """
        Some operations to do after initializing the module.

        :param func: should be a function of satDropout.SatDropout -> satDropout.SatDropout
        :return:
        """
        self.pre_script = func

    def set_params(self, **kwargs):
        """
        Available params:

        -  nsize: (int) the size of instance
        -  dep:(int) the depth of model
        -  maxEpoch: (int) the maximum epoch number
        -  dropt: (int) the times of dropout/tracing
        -  p: (float) the default dropout p value
        -  lr: (float) the learning rate for training
        -  dev: (str) the device name
        -  comm: (str) the comment to store
        :param kwargs:
        :return:
        """
        for nam, val in kwargs.items():
            self.__setattr__(nam, val)
        return self

    def set_psch(self, scheduler=None):
        """
        Set the p-scheduler. If call without param, set as simple lin with p1 = half, p2 = end - 5
        :param scheduler: function of int -> float (0, 1)
        :return: self
        """
        if scheduler is None:
            self.sch = satd.lin_psched(self.dropt // 2, self.dropt - 5)
        else:
            self.sch = scheduler
        return self

    def set_maxEpoch(self, maxEpoch):
        self.maxEpoch = maxEpoch
        return self

    def set_lr(self, lr=1.):
        self.lr = lr
        return self

    def set_p(self, p):
        self.p = p
        return self

    def set_dropt(self, dropt):
        self.dropt = dropt
        return self

    def set_randr(self, randr):
        self.randr = randr
        return self

    def set_clause(self, cls):
        self.cls = cls
        return self

    def set_device(self, device):
        self.dev = device
        return self

    def set_comment(self, comm):
        self.comm = comm

    def add_comment(self, add_comm):
        self.comm += add_comm

    def make_comment(self):
        """
        Make comment add to the current
        :return:
        """
        self.comm += f"\n\n-------- automatically generated --------" \
                     f"\n{'without' if self.sch is None else 'with'} dropout\n" \
                     f"{'without' if not self.randini else 'with'} random init\n" \
                     f"\t rand_range: {self.randr}\n" \
                     f"MaxEpoch: {self.maxEpoch}\tlr: {self.lr}\n" \
                     f"Checkpoint: {self.checkpoint}" \
                     f"\n-------- automatically generated --------\n\n"
        return self

    def rand_on(self, randr):
        self.randini = True
        self.randr = randr
        return self

    def rand_off(self):
        self.randini = False
        self.randr = None
        return self

    def auto_mkdir(self):
        v = f"SAT_tr_n={self.nsize}_{timenow()}"
        try:
            os.mkdir(v)
            return v
        except FileExistsError:
            time.sleep(1)
            return self.auto_mkdir()

    def exportModule(self):
        return self.mod(nsize=self.nsize, clauses=self.cls, dep=self.dep, device=self.dev)

    def run(self, lite=False) -> str:
        """
        will return the destiny folder
        :return:
        """
        module = self.mod(nsize=self.nsize, clauses=self.cls, dep=self.dep, device=self.dev)

        folder = self.auto_mkdir()

        # load param
        # module.load("./SAT_tr_n=12_Oct_12_14:12:01/")
        # module.model.affineDrivenParams(1 / 0.5)
        if self.checkpoint is not None:
            module.load(self.checkpoint)

        if self.pre_script is not None:
            module = self.pre_script(module)

        if self.randini:
            if self.randr is None:
                module.model.randIni()
            else:
                module.model.randIni(*self.randr)

            # Save the initial point
            if not lite:
                torch.save(module.model.state_dict(), f"{folder}/init_state_dict")

        if self.sch is not None:
            module.setPScheduler(self.sch)

        module.run(maxEpoch=self.maxEpoch, dropoutTime=self.dropt, p=self.p, lr=self.lr)


        with open(folder + "/info.json", "w") as f:
            json.dump({
                "Size": self.nsize,
                "ModelDepth": self.dep,
                "ProblemTemplate": self.problemTemplate.__name__,
                "MaxEpoch": self.maxEpoch,
                "LearningRate": self.lr,
                "DropoutTimes": self.dropt,
                "DropoutRatio": self.p,
                "ProblemClause": self.cls
            }, f, indent="\t")
        with open(folder + "/comment.txt", "w") as f:
            f.write(self.comm)
        module.save(folder + "/", lite=lite)

        return folder

    def __call__(self, outfile=None):
        import sys
        temp = sys.stdout

        if outfile is None:
            self.run()
        else:
            with open(outfile, "w") as f:
                sys.stdout = f
                self.run()
            sys.stdout = temp

    def multi_run(self, nproc=8, ntimes=32):
        import multiprocessing as mp

        parf = f"SAT_n=12_tr_rand_collec_{timenow()}"
        os.mkdir(parf)
        os.chdir(parf)
        pool = mp.Pool(nproc)
        i = 0
        for _ in pool.imap(self, ["log_%.3d.txt" % x for x in range(ntimes)]):
            print(f"{i}(/{ntimes})-th run complete.")
            i += 1
        pool.close()
        pool.join()

        os.chdir("..")

        print(f"The collection is saved in {parf}")


class TracingEnv(_envBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mod = satd.trSATD

# Above run_script have passed the test.

if __name__ == "__main__":
    test = TracingEnv()
    _, test_cls = problems.GSATN12C72ClausesPairGet()
    test.set_params(
        dep=2,
        maxEpoch=100,
        dropt=10,
        cls=test_cls[92],
        comm="No. 92, hardest in n=12, with dropout"
    )
    test.set_psch(satd.lin_psched(2, 8))
    test.rand_on((0., 1.))
    test.make_comment()
    test.multi_run(nproc=4, ntimes=4)
