import numpy as np
import matplotlib.pyplot as plt
import os
import re
import torch
import json
from satDropout_New import *
# from satDropout_New import PrefDropoutBase
from SAproc import hammDist, hammNeighbor, strToBarray, barrayToStr


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


class PDistVis:
    def __init__(self, workdir, mod=trSATD):
        # workdir = "SAT_tr_n=12_Wed_Sep_15_16:48:23_2021"
        self.workdir = workdir
        with open(f"{workdir}/info.json", "r") as f:
            c = json.load(f)

        self.module = mod(nsize=c["Size"], clauses=c["ProblemClause"], dep=c["ModelDepth"], device="cpu")
        self.module.load(workdir)

        prob = self.prob = self.module.fullProblem
        self.spec = self.module.fullSpec
        self.tracing = np.load(f"{workdir}/tracing_hist.npy", allow_pickle=True)

        self.sols = self.prob.solution()

        def distToSol(cfg):
            return min([hammDist(strToBarray(cfg), strToBarray(x)) for x in self.sols])

        confInfo = {}
        self.cata = {}


        for i, cfg in enumerate(prob.refcfg):
            dist = distToSol(cfg)
            if dist not in confInfo:
                confInfo.update({dist: [prob(cfg)]})
                self.cata.update({dist: [i]})
            else:
                confInfo[dist].append(prob(cfg))
                self.cata[dist].append(i)


        self.dataX = list(confInfo.keys())
        self.dataY = [np.average(confInfo[x]) for x in self.dataX]
        self.errYmin = [np.average(confInfo[x]) - min(confInfo[x]) for x in self.dataX]
        self.errYmax = [max(confInfo[x]) - np.average(confInfo[x]) for x in self.dataX]

    def plainDraw(self):
        fig, ax = plt.subplots()
        ax.errorbar(self.dataX, self.dataY, yerr=[self.errYmin, self.errYmax], color="blue", fmt="o")
        ax.set_title("Problem Detail")
        ax.set_xlabel("Distance to Solution")
        ax.set_ylabel("Mean Energy", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        return fig, ax

    def draw_probability(self, probVals):
        spec = self.spec
        ps = np.array(probVals)
        meanP = [np.average(ps[x]) for x in list(self.cata.values())]
        fig, ax = self.plainDraw()

        # Add twin axisSAT_tr_n=12_Wed_Sep_15_08:49:43_2021/ as probability
        axp = ax.twinx()
        axp.set_ylabel("Mean Probability", color="red")
        axp.plot(self.dataX, meanP, color="red")
        axp.tick_params(axis="y", labelcolor="red")

        # Add g.s. and 1st e.s. info
        gsInds = self.cata[0]
        esE = sorted(np.unique(spec))[1]
        esInds = [x for x in range(len(spec)) if spec[x] == esE]
        gsX = [0]
        gsY = [np.sum(ps[gsInds])]
        esX = []
        esY = []
        for x, inds in self.cata.items():
            inters = [z for z in inds if z in esInds]
            if len(inters) != 0:
                esX.append(x)
                esY.append(np.sum(ps[inters]))

        axp.bar(gsX, gsY, fc=(1, 1, 0, 0.6))
        axp.bar(esX, esY, fc=(1, 1, 0, 0.6))


        fig.tight_layout()
        return fig

    def get_cumuprob(self, probVal, n: int):
        """
        Return the total probability in the first n eigenstates
        :param n:
        :return:
        """
        return self.prob.get_cumu_probability(probVal, n)
        # spec = self.spec
        # ps = np.array(probVal)
        # elev = sorted(np.unique(spec))
        # reso = min(np.diff(elev))
        # inds = {}
        # for i, x in enumerate(spec):
        #     for k in range(n):
        #         if abs(x - elev[k]) <= reso / 3:
        #             if k not in inds:
        #                 inds[k] = [i]
        #             else:
        #                 inds[k].append(i)
        # v = [0] * n
        # for i, inds in inds.items():
        #     v[i] = np.sum(ps[inds])
        # return v

    def plot_tracing(self):
        # Problem Details Tracing
        cumu = 4
        cumu_data = []
        for x in self.tracing:
            loopat = x['loop']
            fig = self.draw_probability(x["prob"])
            fig.savefig(f"{self.workdir}/probDetails_at_%.3d_loop.png" % loopat)
            plt.close(fig)
            # plt.show()
            cumu_data.append(self.get_cumuprob(x["prob"], cumu))

        # Cumu-probability tracing
        x = list(range(len(cumu_data)))
        cumu_data = np.array(cumu_data)
        fig, ax = plt.subplots()
        for i in range(cumu):
            ax.plot(x, cumu_data[:, i], label=f"elev: {i}")
        ax.set_title("Accumulating Probability")
        ax.set_xlabel("Loop at")
        ax.set_ylabel("Probability")
        ax.legend()
        fig.savefig(f"{self.workdir}/cumu_prob_tracing.png")
        plt.close(fig)

    def plot_lncv(self):
        learn_curv, brkpt = readLoss(self.workdir)
        fig, ax = plt.subplots()
        ax.plot(learn_curv)
        for x in brkpt:
            ax.axvline(x=x, linestyle="--", color="red")
        ax.set_title("learning curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("E loss")
        fig.savefig(f"{self.workdir}/learn_curv.png")
        plt.close(fig)


def plotDropoutSpec(vis: PDistVis):
    confInfo = {}
    cata = {}
    module = vis.module
    prob = module.fullProblem

    sols = prob.solution()
    spec = module.model.lays[0].spec.cpu().numpy()

    def distToSol(cfg):
        return min([hammDist(strToBarray(cfg), strToBarray(x)) for x in sols])

    for i, cfg in enumerate(prob.refcfg):
        dist = distToSol(cfg)
        if dist not in confInfo:
            confInfo.update({dist: [spec[i]]})
            cata.update({dist: [i]})
        else:
            confInfo[dist].append(spec[i])
            cata[dist].append(i)

    dataX = list(confInfo.keys())
    dataY = [np.average(confInfo[x]) for x in dataX]
    errYmin = [np.average(confInfo[x]) - min(confInfo[x]) for x in dataX]
    errYmax = [max(confInfo[x]) - np.average(confInfo[x]) for x in dataX]

    fig, ax = plt.subplots()
    ax.errorbar(dataX, dataY, yerr=[errYmin, errYmax], color="blue", fmt="o")
    ax.errorbar(np.array(vis.dataX) + 0.2, vis.dataY, yerr=[vis.errYmin, errYmax], color="red", fmt="o")
    ax.axhline(y=0., xmin=0, xmax=max(dataX)+0.5)
    ax.set_title("Problem Detail")
    ax.set_xlabel("Distance to Solution")
    ax.set_ylabel("Mean Energy", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    return fig, ax


def saveInfoAsText(vis: PDistVis, folder: str):
    with open(f"{folder}/problemInfo.txt", "w") as f:
        for cfg, ener in zip(vis.prob.refcfg, vis.prob.spec):
            f.write(f"{''.join(cfg)}\t{ener}\n")

    probs = [x["prob"] for x in vis.tracing]
    np.savetxt(f"{folder}/tracing.txt", probs)

    cumu_data = []
    for x in vis.tracing:
        loopat = x['loop']
        fig = vis.draw_probability(x["prob"])
        fig.savefig(f"{vis.workdir}/probDetails_at_%.3d_loop.png" % loopat)
        plt.close(fig)
        # plt.show()
        cumu_data.append(vis.get_cumuprob(x["prob"], 10))

    np.savetxt(f"{folder}/cumu_prob_10.txt", cumu_data)


if __name__ == "__main__":
    # folder = "DataCollec/n=16/CrossLoss/easy00-hard25-doub/"
    folder = "DataCollec/n=16/CrossLossFailed/easy00-hard25/"
    # folder = "DataCollec/n=16/NeoC112_025/diffLay@origin/Depth=030/"
    vis = PDistVis(folder, mod=PrefDropoutBase)
    saveInfoAsText(vis, folder)

    vis.plot_lncv()
    vis.plot_tracing()

    fig, ax = plotDropoutSpec(vis)
    fig.savefig(f"{vis.workdir}/spec_ref.png")
    plt.close("all")
    #  plt.show()