import qnn
import problems
import torch
import math
import numpy as np
import random
import json
import sys
import re
import itertools

from io import StringIO
random.seed(0)

# problemTemplate = problems.OneInThreeSAT
# problemTemplate = problems.HardGeneralSAT
problemTemplate = problems.HardGeneralSAT_normalized


class DynaEvoIsingLayer(qnn.EvoIsingHamiltonian):
    def __init__(self, *args):
        super().__init__(*args)

    def modSpec(self, _spec):
        self.register_buffer("spec", torch.tensor(_spec, dtype=torch.float, device=self.spec.device))


class DropOutQAOA(torch.nn.Module):
    def __init__(self, size, spec, dep=3):
        super().__init__()
        self.spec = spec
        self.size = size
        self.dept = dep
        self.lays = torch.nn.ModuleList()
        self.drvLays = []
        ind = 0
        for x in range(dep):
            t = (x + 1) / (dep + 1)
            self.lays.append(DynaEvoIsingLayer(self.size, self.spec, t))
            self.drvLays.append(ind)
            self.lays.append(qnn.EvoXMixing(self.size, 1 - t))
            # self.lays.append(qnn.EvoXMixingNormalized(self.size, 1 - t))
            ind += 2

        self.defaultParam = self.getParams()
        self.defaultT = self.getDrivenParams()

    def tReset(self):
        self.loadDrivenParams(self.defaultT)

    def reset(self):
        self.loadParams(self.defaultParam)

    def forward(self, x):
        for layer in range(len(self.lays)):
            x = self.lays[layer](x)
        return x

    def specSetup(self, specs):
        for i, pos in enumerate(self.drvLays):
            self.lays[pos].modSpec(specs[i])

    def specReset(self):
        for i, pos in enumerate(self.drvLays):
            self.lays[pos].modSpec(self.spec)

    def affineDrivenParams(self, factor):
        # state_dic = self.state_dict()
        # for lay in self.drvLays:
        #     state_dic[f"lays.{lay}.t"] *= factor
        # self.load_state_dict(state_dic)
        now = self.getDrivenParams()
        for i in range(len(now)):
            now[i] *= factor
        self.loadDrivenParams(now)

    def getDrivenParams(self):
        state_dic = self.state_dict()
        v = []
        for lay in self.drvLays:
            v.append(state_dic[f"lays.{lay}.t"])
        return v

    def loadDrivenParams(self, vals):
        state_dic = self.state_dict()
        for lay, v in zip(self.drvLays, vals):
            state_dic[f"lays.{lay}.t"] = v
        self.load_state_dict(state_dic)

    def loadParams(self, vals):
        state_dic = self.state_dict()
        for lay, v in zip(range(2 * self.dept), vals):
            param_key = f"lays.{lay}.t" if lay % 2 == 0 else f"lays.{lay}.evolu.t"
            state_dic[param_key] = v
        self.load_state_dict(state_dic)

    def getParams(self):
        state_dic = self.state_dict()
        v = []
        for lay in range(2 * self.dept):
            param_key = f"lays.{lay}.t" if lay % 2 == 0 else f"lays.{lay}.evolu.t"
            v.append(state_dic[param_key])
        return v

    def randIni(self, start=0., end=1.):
        # Generating a random float array, with length of 2 * self.dept
        vals = [random.uniform(start, end) for _ in range(2 * self.dept)]

        vals = torch.tensor(vals)
        self.loadParams(vals)
        self.defaultParam = self.getParams()


class SATManual:
    def __init__(self, nsize, clauses=None, dep=3, device="cpu", probT=problems.HardGeneralSAT_normalized):
        self.size = nsize

        if clauses is None:
            self.clauses = []
        else:
            self.clauses = clauses

        self.problemTemplate = probT
        self.fullProblem = self.problemTemplate(nsize, self.clauses)
        _, self.fullSpec = self.fullProblem.info()
        self._dev_name = device
        self.device = torch.device(device)

        self.depth = dep
        self.model = DropOutQAOA(nsize, self.fullSpec, dep).to(self.device)
        self.critr = qnn.ExpectValue(nsize, self.fullSpec).to(self.device)
        self.log = self.genEmptyLog()
        self.psched = None

        self.reg_subspecs = {}
        self.optim = None

    def genEmptyLog(self):
        return {
            "learn_curv": [""],
            "misc": {},
            "tracing_hist": []
        }

    def reset(self):
        self.log = self.genEmptyLog()
        self.model.reset()

    def setOptimizer(self, optimizer_func):
        self.optim = optimizer_func(self.model.parameters())

    def setFullProblem(self, spec):
        self.fullSpec = spec
        self.fullProblem = problems._spinGlassProblem(self.size)
        self.fullProblem.spec = spec
        self.setLoss(spec)

    def setLoss(self, spec):
        self.critr = qnn.ExpectValue(self.size, spec).to(self.device)

    def setLayerSpec(self, specs):
        """
        Set the model with given specs

        :param specs: is the list with len == self.depth. specs[i] should be the spec used at i-th layer
        :type specs: list
        :return:
        """
        self.model.specSetup(specs)

    def _OptimEval(self, optimizer, maxEpoch):
        for t in range(maxEpoch):
            x = 1. / math.sqrt(2 ** self.size) * torch.ones(1, 2 ** self.size, device=self.device)

            def closure():
                optimizer.zero_grad()
                y = self.model(x)
                eloss = self.critr(y)
                eloss.backward()
                return eloss

            optimizer.step(closure)

            with torch.no_grad():
                y = self.model(x)
                eloss = self.critr(y)

            deltaE = eloss.item()
            prob = np.abs(y.cpu().numpy()) ** 2
            full_los = self.fullProblem.get_energy_loss(prob)
            if maxEpoch < 5 or (t > 1 and t % int(maxEpoch / 5) == 0):
                print(f"\tEpoch-{t}: Curr_loss = {deltaE}, Full_loss = {full_los}")

            self.log["learn_curv"][-1] += f"{t}\t{deltaE}\n"

        self.log["learn_curv"].append("")

    def train_step(self, **internalStates):
        """
        One step of the training
        :param kwargs: optional parameter setup:
            -  subtim (int): training times in one step
        :return: None
        """
        subtim = internalStates["subtim"]
        self._OptimEval(self.optim, subtim)

    def perf_train(self, **internalStates):
        subtim = internalStates["subtim"]
        outLoop = internalStates["outLoop"]
        if subtim > 1:
            print(f"------------- Begin {outLoop}-th Loop -------------")
            # if p < 1. - 1e-6:
            #     nspec = self.genDropoutSpec(p)
            #     self.model.specSetup([nspec] * self.depth)
            # self.LBFGSeval(subtim, lr)
            self.train_step(**internalStates)
            print(f"-------------- End {outLoop}-th Loop --------------")
        else:
            stdout_temp = sys.stdout
            sys.stdout = myout = StringIO()
            self.train_step(**internalStates)
            sys.stdout = stdout_temp

            res = myout.getvalue()
            res = res.replace('\n', '')
            print(f"Loop-{outLoop}: {res}")

    def test_final(self, **internalStates):
        """
        At least to compute the final success probability and return
        :return:
        """
        _, prob = self.test_step(**internalStates)
        print("Final energy loss:", self.fullProblem.get_energy_loss(prob))
        succp = self.fullProblem.get_cumu_probability(prob, 1)
        print(f"Success Probability: {succp}")
        self.log["misc"]["Success Probability"] = succp
        return succp

    def test_step(self, **internalStates):
        x = 1. / math.sqrt(2 ** self.size) * torch.ones(1, 2 ** self.size, device=self.device)
        with torch.no_grad():
            y = self.model(x)
        prob = y.detach().cpu().numpy()
        prob = (np.abs(prob) ** 2).squeeze()
        return self.fullSpec, prob

    def perf_test(self, prev_weight, **internalStates):
        outLoop = internalStates["outLoop"]
        s = prev_weight
        _, prob = self.test_step(**internalStates)
        ts = self.model.getParams()

        ts = np.array([x.cpu() for x in ts])

        # print d_param and the occupation of first three level
        print(f"\td_param={np.linalg.norm(ts - s)}\tsucc_p={self.fullProblem.get_cumu_probability(prob, 3)}")
        s = ts

        # Convert to built-in types
        self.log["tracing_hist"].append({"loop": outLoop, "prob": list(prob), "t": [float(x) for x in ts]})

        return s

    def preScript(self, **internalStates) -> dict:
        return {}

    def postScript(self, **internalStates):
        return

    def run(self, maxEpoch: int, dropoutTime: int, **internalStates):
        subtim = int(maxEpoch / dropoutTime)

        s = np.ones(2 * self.depth)
        for outLoop in range(dropoutTime):
            internalStates.update({"outLoop": outLoop, "subtim": subtim})
            internalStates.update(self.preScript(**internalStates))
            self.perf_train(**internalStates)
            s = self.perf_test(s, **internalStates)
            self.postScript(**internalStates)

        return self.test_final(**internalStates)

    def savelog(self, prefix):
        for i, log in enumerate(self.log["learn_curv"]):
            # Skip empty log
            if len(log) == 0:
                continue
            with open(prefix + ("/epoch_eloss_%.3d.txt" % i), "w") as f:
                f.write(log)
        with open(f"{prefix}/misc.txt", "w") as f:
            for key, val in self.log["misc"].items():
                f.write(f"{key}:\t{val}\n")

        np.save(prefix + "tracing_hist", np.array(self.log["tracing_hist"], dtype=object))

    def genSavingJSON(self):
        return {
            "size": self.size,
            "depth": self.depth,
            "clauses": self.clauses,
            "fullSpec": list(self.fullSpec)
        }

    def save(self, prefix, lite=False):
        self.savelog(prefix)
        with open(f"{prefix}/module.json", "w") as f:
            json.dump(self.genSavingJSON(), f)

        if not lite:
            torch.save(self.model.state_dict(), f"{prefix}/model_state_dict")
        print(f"Model saved in dir {prefix}")

    def load(self, prefix):
        with open(f"{prefix}/module.json", "r") as f:
            state = json.load(f)
        self.__init__(nsize=state["size"], clauses=state["clauses"], dep=state["depth"], device=self._dev_name)
        try:
            self.model.load_state_dict(torch.load(f"{prefix}/model_state_dict"))
        except:
            pass


class SATDropOut(SATManual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.psched = None

    def setPScheduler(self, ps=None):
        """
        ps should be a function integer -> (0, 1), as the p value at each dropout stage.
        :param ps:
        :return:
        """
        self.psched = ps

    def genDropoutSpec(self, p: float):
        diff = 0
        while diff == 0:
            newcls = random.sample(self.clauses, int(p * len(self.clauses)))
            prb = self.problemTemplate(self.size, newcls)
            size, spec = prb.info()
            diff = max(spec) - min(spec)
        return spec

    def train_step(self, **internalStates):
        """
        Single step of the training
        :param p: float: the dropout probability
        :param lr: float: the learning rate
        :param subtim: int: the number of sub epochs.
        :return: None
        """
        p = internalStates["p"]
        subtim = internalStates["subtim"]

        if p < 1. - 1e-6:
            nspec = self.genDropoutSpec(p)
        else:
            nspec = self.fullSpec

        self.model.specSetup([nspec] * self.depth)
        self._OptimEval(self.optim, subtim)

    def test_step(self, **internalStates):
        p = internalStates["p"]
        self.model.affineDrivenParams(p)
        res = super().test_step(**internalStates)
        self.model.affineDrivenParams(1 / p)
        return res

    def preScript(self, **internalStates) -> dict:
        if self.psched is not None:
            p = self.psched(internalStates["outLoop"])
        elif "p" in internalStates:
            p = internalStates["p"]
        else:
            p = 1.
        return {"p": p}

    def postScript(self, **internalStates):
        outLoop = internalStates["outLoop"]
        p = internalStates["p"]
        self.log["misc"][f"loop_{outLoop}_pval"] = p

    def run(self, maxEpoch: int, dropoutTime: int, **internalStates):
        res = super().run(maxEpoch=maxEpoch, dropoutTime=dropoutTime, **internalStates)
        self.model.affineDrivenParams(internalStates["p"])
        return res

    def genSavingJSON(self):
        return {
                "size": self.size,
                "depth": self.depth,
                "clauses": self.clauses
        }

    def load(self, prefix):
        super().load(prefix)
        # restore the state of training: reaffine p
        with open(f"{prefix}/misc.txt", "r") as f:
            misc = f.readlines()
        p_fin_lin = [r for r in misc if "loop" in r][-1]
        p_fin = float(re.findall(r".*\t([\d\.]+).*", p_fin_lin)[-1])
        self.model.affineDrivenParams(1. / p_fin)


class trSATD_DynaLoss(SATDropOut):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, **internalStates):
        p = internalStates["p"]
        subtim = internalStates["subtim"]

        if p < 1. - 1e-6:
            nspec = self.genDropoutSpec(p)
        else:
            nspec = self.fullSpec

        self.model.specSetup([nspec] * self.depth)
        self.critr = qnn.ExpectValue(self.size, nspec).to(self.device)

        self._OptimEval(self.optim, subtim)


class trSATD_DynaLoss_FxiedModel(SATManual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setPScheduler(self, ps=None):
        """
        ps should be a function integer -> (0, 1), as the p value at each dropout stage.
        :param ps:
        :return:
        """
        self.psched = ps

    def genDropoutSpec(self, p: float):
        diff = 0
        while diff == 0:
            newcls = random.sample(self.clauses, int(p * len(self.clauses)))
            prb = self.problemTemplate(self.size, newcls)
            size, spec = prb.info()
            diff = max(spec) - min(spec)
        return spec

    def preScript(self, **internalStates) -> dict:
        if self.psched is not None:
            p = self.psched(internalStates["outLoop"])
        return {"p": p}

    def postScript(self, **internalStates):
        outLoop = internalStates["outLoop"]
        p = internalStates["p"]
        self.log["misc"][f"loop_{outLoop}_pval"] = p

    def train_step(self, **internalStates):
        p = internalStates["p"]
        subtim = internalStates["subtim"]

        if p < 1. - 1e-6:
            nspec = self.genDropoutSpec(p)
        else:
            nspec = self.fullSpec

        # self.model.specSetup([nspec] * self.depth)
        self.setLoss(nspec)
        self._OptimEval(self.optim, subtim)

    def genSavingJSON(self):
        return {
                "size": self.size,
                "depth": self.depth,
                "clauses": self.clauses
        }


class trSATD(SATDropOut):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optim = torch.optim.LBFGS(self.model.parameters(), lr=1e-2)

    def preScript(self, **internalStates) -> dict:
        self.optim = torch.optim.LBFGS(self.model.parameters(), lr=1e-2)
        return super().preScript(**internalStates)


class PrefDropoutBase(SATManual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ess = []
        self.nes = []

    def setEssential(self, clauses):
        for cls in self.clauses:
            if cls in clauses:
                self.ess.append(cls)
            else:
                self.nes.append(cls)

    def genDropoutSpec(self, p: float):
        diff = 0
        while diff == 0:
            newcls = random.sample(self.nes, int(p * len(self.nes)))
            newcls += self.ess
            prb = self.problemTemplate(self.size, newcls)
            size, spec = prb.info()
            diff = max(spec) - min(spec)
        return spec


#  Optional name
# trSATD = SATDropOut
SATSGD = trSATD_DynaLoss_FxiedModel
trSATD_Manual_FixedModel = SATManual


class lin_psched:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, i):
        p1 = self.p1
        p2 = self.p2
        if i < p1:
            return 0.5
        elif i < p2:
            return 0.5 + (1. - 0.5) / (p2 - p1) * (i - p1)
        return 1.


# def lin_psched(p1, p2):
#     """
#     Stupidly simple p val scheduling
#     :param i:
#     :return:
#     """
#     def sch(i):
#         if i < p1:
#             return 0.5
#         elif i < p2:
#             return 0.5 + (1. - 0.5) / (p2 - p1) * (i - p1)
#         return 1.
#     return sch
