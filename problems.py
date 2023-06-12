import numpy as np
from qnn import _hamil2spec
import string
from typing import Dict, List, Union, Tuple

import mkl
mkl.set_num_threads(1)


class _spinGlassProblem:
    def __init__(self, size):
        self.size = size
        self.spec = np.zeros(2 ** size)
        self.hamil = {
            "z": [0.] * self.size,
            "zz": {(i, j): 0. for i in range(self.size) for j in range(i+1, self.size)}
        }
        self.refcfg = {x: i for i, x in enumerate(self._refConf())}
        self.normalized = False

    def info(self):
        return self.size, self.spec

    def solution(self):
        confs = np.array(list(self.refcfg.keys()))
        idl = self.spec == min(self.spec)
        return confs[idl]

    def _refConf(self):
        return [np.binary_repr(x, self.size) for x in range(2 ** self.size)]

    def __str__(self):
        confs = list(self.refcfg.keys())
        idl = np.argsort(self.spec)
        return "\n".join([str((confs[x], self.spec[x])) for x in idl])

    def savedata(self, filename):
        confs = list(self.refcfg.keys())
        with open(filename, "w") as f:
            for x in np.argsort(self.spec):
                f.write(f"{confs[x]} {self.spec[x]}\n")

    def level_detail(self, n: int):
        confs = list(self.refcfg.keys())
        elev, reso = self._levels_and_min_spc()
        ener = elev[n]
        idl = np.arange(len(self.spec))
        msk = np.abs(self.spec - ener) < reso
        return ener, [confs[x] for x in idl[msk]]

    def __call__(self, x):
        if isinstance(x, str):
            f = x
        else:
            f = "".join([str(z) for z in x])
            if f not in self.refcfg:
                print(f)
                raise ValueError("Invalid call for prob.spec")
        return self.spec[self.refcfg[f]]

    def cutoff(self, cutof):
        self.spec_origin = self.spec
        self.spec = np.clip(self.spec, 0, cutof)

    def resetSpec(self):
        self.spec = self.spec_origin

    def normalize_spec(self):
        self.spec_origin = self.spec.copy()
        self.normalized = True
        if max(self.spec) == min(self.spec):
            self.spec = np.zeros(len(self.spec))
        else:
            self.spec /= (max(self.spec) - min(self.spec))

    def isNormalized(self):
        return self.normalized

    def _levels_and_min_spc(self):
        elev = sorted(np.unique(self.spec))
        reso = min(np.diff(elev))
        return elev, reso/2

    def _levels_indices(self, n=None):
        spec = self.spec
        elev = sorted(np.unique(spec))
        reso = min(np.diff(elev))
        if n is None:
            _n = len(elev)
        else:
            _n = min(n, len(elev))
        inds = {}
        for i, x in enumerate(spec):
            for k in range(_n):
                if abs(x - elev[k]) <= reso / 3:
                    if k not in inds:
                        inds[k] = [i]
                    else:
                        inds[k].append(i)
        v = [0] * _n
        for i, inds in inds.items():
            v[i] = (elev[i], inds)
        return v, len(v)

    def cumu_prob_pair_getter(self):
        v, _n = self._levels_indices()
        def getter(probVal):
            ps = np.array(probVal)
            res = [0] * _n
            for i in range(_n):
                res[i] = (v[i][0], np.sum(ps[v[i][1]]))

            return res
        return getter

    def get_cumu_probability_pair(self, probVal, n=None):
        """
        Return the total probability in pair form: [(E, P(E)), ...]

        :param probVal:
        :return:
        """
        v, _n = self._levels_indices(n)
        ps = np.array(probVal)
        res = [0] * _n
        for i in range(_n):
            res[i] = (v[i][0], np.sum(ps[v[i][1]]))

        return res

        # spec = self.spec
        # ps = np.array(probVal)
        # elev = sorted(np.unique(spec))
        # reso = min(np.diff(elev))
        # if n is None:
        #     _n = len(elev)
        # else:
        #     _n = n
        # inds = {}
        # for i, x in enumerate(spec):
        #     for k in range(_n):
        #         if abs(x - elev[k]) <= reso / 3:
        #             if k not in inds:
        #                 inds[k] = [i]
        #             else:
        #                 inds[k].append(i)
        # v = [0] * _n
        # for i, inds in inds.items():
        #     v[i] = (elev[i], np.sum(ps[inds]))
        # return v

    def get_cumu_probability(self, probVal, n: int):
        """
        Return the total probability in the first n eigenstates
        :param n:
        :return:
        """
        v = self.get_cumu_probability_pair(probVal, n)[:n]
        return [x[1] for x in v]
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

    def get_energy_loss(self, probVal: np.ndarray):
        return np.dot(probVal, self.spec)


class MaxCut(_spinGlassProblem):
    def __init__(self, graphMat):
        super().__init__(len(graphMat))
        self.graph = graphMat
        for (i, j) in self.hamil["zz"]:
            self.hamil["zz"][(i, j)] = - self.graph[i, j]
        self.spec = _hamil2spec(self.hamil)
        self.spec -= min(self.spec)


class OneInThreeSAT(_spinGlassProblem):
    def __init__(self, _size, _bolist):
        super().__init__(_size)
        for clause in _bolist:
            x, y, z = sorted(clause)
            for i in [x, y, z]:
                self.hamil["z"][i] -= 1
            for (i, j) in [(x, y), (x, z), (y, z)]:
                self.hamil["zz"][(i, j)] += 1
        self.spec = _hamil2spec(self.hamil)
        self.spec -= min(self.spec)


class NumberPartition(_spinGlassProblem):
    def __init__(self, _size, _numblist):
        super().__init__(_size)
        for i, ni in enumerate(_numblist):
            for j, nj in enumerate(_numblist):
                if j > i:
                    self.hamil["zz"][(i, j)] += nj * ni
        self.spec = _hamil2spec(self.hamil)
        # renormalization
        # self.spec = (self.spec - min(self.spec)) / (max(self.spec) - min(self.spec))
        # cutoff
        # self.spec = np.clip(self.spec - min(self.spec), 0, 100)
        self.spec = self.spec - min(self.spec)


class HardOneInThreeSAT(_spinGlassProblem):
    def __init__(self, _size, _bolist):
        super().__init__(_size)
        for clause in _bolist:
            x, y, z = clause
            self.hamil["z"][x] += 1
            self.hamil["z"][y] += 1
            self.hamil["z"][z] -= 1

            self.hamil["zz"][tuple(sorted((x, y)))] += 1
            self.hamil["zz"][tuple(sorted((y, z)))] -= 1
            self.hamil["zz"][tuple(sorted((z, x)))] -= 1
        self.spec = _hamil2spec(self.hamil)
        self.spec -= min(self.spec)


# For Frank's Clauses
class HardGeneralSAT(_spinGlassProblem):
    def __init__(self, _size, _bolist):
        super().__init__(_size)
        for clause in _bolist:
            x, y, z = clause
            self.hamil["zz"][tuple(sorted((x, y)))] += 1
            self.hamil["zz"][tuple(sorted((y, z)))] += 1
            self.hamil["zz"][tuple(sorted((z, x)))] += 1
        self.spec = _hamil2spec(self.hamil)
        self.spec -= min(self.spec)

        self.cls = _bolist

    def vio_clauses(self, conf: Union[str, list, np.ndarray]) -> Dict[str, List[int]]:
        """
        Split original clauses into two parts:
            - "agr": conf agrees
            - "vio": conf violates

        :param conf: configuration, as 01 string or list(np.ndarray), with 0/1 elements
        :return:
        """
        agreed = []
        disagr = []
        if isinstance(conf, str):
            x = np.array([int(z) for z in conf])
        elif isinstance(conf, list):
            x = np.array(conf)
        elif isinstance(conf, np.ndarray):
            x = conf
        else:
            raise ValueError("conf should be str or list")
        for i in range(len(self.cls)):
            cls = self.cls[i]
            if sum(x[cls]) not in [0, 3]:
                agreed.append(i)
            else:
                disagr.append(i)
        return {"agr": agreed, "vio": disagr}

    def essential_clauses_by_given_confs(self, confs):
        res = []
        for x in confs:
            res += self.vio_clauses(x)["vio"]
        return [self.cls[i] for i in set(res)]

    def essential_clauses(self, n: int):
        """
        Find the essential clauses for the first n level (except ground energy)

        :param n:
        :return:
        """
        confs = []
        for i in range(1, 1+n):
            confs += self.level_detail(i)[1]
        return self.essential_clauses_by_given_confs(confs)


# For Frank's Clauses, but normalized for better optim
class HardGeneralSAT_normalized(HardGeneralSAT):
    def __init__(self, _size, _bolist):
        super().__init__(_size, _bolist)
        self.normalize_spec()
        # self.spec /= (max(self.spec) - min(self.spec))


class N16PrefGetter:
    def __init__(self, prefix="./N16PrefNeo/N16C101hardest", lowconfs=True, rdLen=200):
        cls = []
        conf = []
        succ = []
        self.len = rdLen
        with open(f"{prefix}.npy", "rb") as f:
            for _ in range(self.len):
                cl = np.load(f, allow_pickle=False)
                cl = [[int(y) for y in x] for x in cl]
                cls.append(cl)
        if lowconfs:
            with open(f"{prefix}pmfconf.npy", "rb") as f:
                for _ in range(self.len):
                    cl = np.load(f, allow_pickle=False)
                    # make it to list, fix the bug of random.shuffle(np.ndarray)
                    conf.append([list(x) for x in np.int_((cl + 1) / 2)])
        else:
            conf = [None] * self.len
        with open(f"{prefix}_notes.txt", "r") as f:
            for lin in f.readlines():
                ind, trialnum, succnum, lownum = [int(x) for x in lin.split() if x.isdigit()]
                succ.append(succnum / trialnum)

        self.cls = cls
        self.conf = conf
        self.succ = succ
        self.get = self.__getitem__

    def __len__(self):
        return self.len

    def __getitem__(self, item: int) -> dict:
        """
        return the items

        :param item: int as the index
        :return: dict with keys: "clause", "conf", "succ_p"
        """
        return {
            "clause": self.cls[item],
            "conf": self.conf[item],
            "succ_p": self.succ[item]
        }

    def get_hardest(self, n: int) -> List[dict]:
        inds = np.argsort(self.succ)[:n]
        return [self[x] for x in inds]

    def get_easiest(self, n: int) -> Tuple[list, list, list]:
        inds = np.argsort(self.succ)[-n:]
        return [self[x] for x in inds]


GSATN16C101Getter = N16PrefGetter("./N16PrefNeo/N16C101hardest", lowconfs=True, rdLen=200)
GSATN16C112Getter = N16PrefGetter("./N16PrefNeo/N16C112hardest", lowconfs=True, rdLen=200)
GSATN16C101EasyGetter = N16PrefGetter("./N16PrefNeo/N16C101easy", lowconfs=False, rdLen=2)


#######################
# Use this to replace the old getter
#######################
#
# GSATN16C128EasyClause = []
# GSATN16C128HardClause = []
# with open("./N16C128Clauses/N16C128easy.npy", "rb") as f:
#     for _ in range(100):
#         cl = np.load(f, allow_pickle=False)
#         cl = [[int(y) for y in x] for x in cl]
#         GSATN16C128EasyClause.append(cl)
# with open("./N16C128Clauses/N16C128hard.npy", "rb") as f:
#     for _ in range(100):
#         cl = np.load(f, allow_pickle=False)
#         cl = [[int(y) for y in x] for x in cl]
#         GSATN16C128HardClause.append(cl)
#
def GSATN16C128ClausesPairGet():
    """
    Get the easy and hard clauses for n16c128.

    It returns a tuple as (easy_clauses, hard_clauses)

    :return:
    """
    GSATN16C128EasyClause = []
    GSATN16C128HardClause = []
    with open("./N16C128Clauses/N16C128easy.npy", "rb") as f:
        for _ in range(100):
            cl = np.load(f, allow_pickle=False)
            cl = [[int(y) for y in x] for x in cl]
            GSATN16C128EasyClause.append(cl)
    with open("./N16C128Clauses/N16C128hard.npy", "rb") as f:
        for _ in range(100):
            cl = np.load(f, allow_pickle=False)
            cl = [[int(y) for y in x] for x in cl]
            GSATN16C128HardClause.append(cl)

    return GSATN16C128EasyClause, GSATN16C128HardClause



#######################
# Old implementation
#######################
#
# GSATN16C128HardestClause = []
# with open("./N16C128Clauses/N16C128hardest.npy", "rb") as f:
#     while True:
#         try:
#             cl = np.load(f)
#             cl = [[int(y) for y in x] for x in cl]
#             GSATN16C128HardestClause.append(cl)
#         except:
#             break
#
# GSATN16C128HardestInfo = []
# with open("./N16C128Clauses/N16C128hardest_notes.txt", "r") as f:
#     ref_p = 1.
#     min_ind = 0
#     ind = 0
#     for lin in f.readlines():
#         num, tot, succ = [int(float(x)) for x in lin.split() if '0' in x or x.isdigit()]
#         succ_p = succ / tot
#         if succ_p > 0.15:
#             continue
#         else:
#             if succ_p < ref_p:
#                 ref_p = succ_p
#                 min_ind = ind
#             GSATN16C128HardestInfo.append({
#                 "old_index": num,
#                 "succ_p": succ / tot
#             })
#             ind += 1
#     GSATN16C128HardestInstance, GSATN16C128HardestInstanceSuccP = GSATN16C128HardestClause[min_ind], ref_p
#
#
def GSATN16C128HardestGet():
    """

    return the tuple as (clauses, info, hardest_instance, hardest_succp)

    :return:
    """
    GSATN16C128HardestClause = []
    with open("./N16C128Clauses/N16C128hardest.npy", "rb") as f:
        while True:
            try:
                cl = np.load(f)
                cl = [[int(y) for y in x] for x in cl]
                GSATN16C128HardestClause.append(cl)
            except:
                break

    GSATN16C128HardestInfo = []
    with open("./N16C128Clauses/N16C128hardest_notes.txt", "r") as f:
        ref_p = 1.
        min_ind = 0
        ind = 0
        for lin in f.readlines():
            num, tot, succ = [int(float(x)) for x in lin.split() if '0' in x or x.isdigit()]
            succ_p = succ / tot
            if succ_p > 0.15:
                continue
            else:
                if succ_p < ref_p:
                    ref_p = succ_p
                    min_ind = ind
                GSATN16C128HardestInfo.append({
                    "old_index": num,
                    "succ_p": succ / tot
                })
                ind += 1
        GSATN16C128HardestInstance, GSATN16C128HardestInstanceSuccP = GSATN16C128HardestClause[min_ind], ref_p

    return GSATN16C128HardestClause, GSATN16C128HardestInfo, GSATN16C128HardestInstance, GSATN16C128HardestInstanceSuccP


######################
# Old implementation
######################
#
# GSATN12C72EasyClause = []
# GSATN12C72HardClause = []
# with open("./N12C72Clauses/N12C72easy.npy", "rb") as f:
#     for _ in range(100):
#         cl = np.load(f, allow_pickle=False)
#         cl = [[int(y) for y in x] for x in cl]
#         GSATN12C72EasyClause.append(cl)
# with open("./N12C72Clauses/N12C72harder.npy", "rb") as f:
#     for _ in range(100):
#         cl = np.load(f, allow_pickle=False)
#         cl = [[int(y) for y in x] for x in cl]
#         GSATN12C72HardClause.append(cl)
#
def GSATN12C72ClausesPairGet():
    """

    return the tuple as (easy_clauses, hard_clauses)

    :return:
    """
    GSATN12C72EasyClause = []
    GSATN12C72HardClause = []
    with open("./N12C72Clauses/N12C72easy.npy", "rb") as f:
        for _ in range(100):
            cl = np.load(f, allow_pickle=False)
            cl = [[int(y) for y in x] for x in cl]
            GSATN12C72EasyClause.append(cl)
    with open("./N12C72Clauses/N12C72harder.npy", "rb") as f:
        for _ in range(100):
            cl = np.load(f, allow_pickle=False)
            cl = [[int(y) for y in x] for x in cl]
            GSATN12C72HardClause.append(cl)

    return GSATN12C72EasyClause, GSATN12C72HardClause


ex0 = OneInThreeSAT(4, [[0, 1, 2]])
ex1 = OneInThreeSAT(4, [[0, 1, 2], [1, 2, 3]])
ex2 = OneInThreeSAT(4, [[0, 1, 2], [1, 2, 3], [1, 0, 3]])

ex_test_easy = OneInThreeSAT(8, [[0, 1, x] for x in range(4, 8)] + [[2, 3, x] for x in range(4, 8)] + [[0, 1, 4]])
ex_test_hard = OneInThreeSAT(8, [[0, 1, x] for x in range(4, 8)] + [[2, 3, x] for x in range(4, 8)] + [[0, 1, 2]])

nbp = NumberPartition(8, [24, 25, 26, 13, 14, 15, 16, 17])
nbp_easy8 = NumberPartition(8, [14, 15, 12, 13, 14, 15, 16, 17])
nbp_tiny = NumberPartition(5, [4, 5, 6, 7, 8])
nbp_easy = NumberPartition(5, [1, 2, 3, 4, 4])
nbp_4 = NumberPartition(4, [1, 1, 1, 3])


def gen1in3SATClausePair(nsize):
    """
    generate the hard/easy instance of 1in3SAT problem, used for OneInThreeSAT problem
    :param nsize: system size, int
    :return:
        -  hard_cls - clauses for hard instance, list
        -  easy_cls - clauses for easy instance, list
    """
    if nsize < 4:
        raise ValueError("There is no available instances")
    p1 = [0, 1]
    p2 = [2, 3]
    easy_cls = [p1 + [x] for x in range(4, nsize)] + [p2 + [x] for x in range(4, nsize)]
    hard_cls = easy_cls + [[0, 1, 2]]
    return hard_cls, easy_cls


def Legacy_genHard1in3SATClausePair(nsize):
    """
    generate the hard/easy instance of 1in3SAT problem
    :param nsize: system size, int
    :return nsize: system size, int
    :return hardCls: clauses for hard instance, HardOneInThreeSAT
    :return easyCls: clauses for easy instance, HardOneInThreeSAT
    """
    baseCls = [[i, (i + 1) % nsize, (i + 2) % nsize] for i in range(nsize)]
    added = []
    easyd = []
    for i in range(1, nsize - 1):
        easyd.extend([
            [0, i, nsize-1], [i, 0, nsize-1], [nsize - 1, i, 0], [i, nsize - 1, 0]
        ])
        for j in range(1, nsize - 1):
            if i != j:
                added.extend([
                    [0, i, j], [i, 0, j], [nsize - 1, i, j], [i, nsize - 1, j]
                ])


    hardCls = baseCls + added
    # does easyd a easy case? (which makes the first excited be 0111...10
    # easyCls = baseCls + easyd
    easyCls = baseCls
    return nsize, hardCls, easyCls


def Legacy_genHard1in3SATPair(nsize):
    """
    Convert the hard/easy 1in3SAT clause to problem struct, normalized energy loss (by make |C| be equal).

    :param nsize: system size
    :return: hard, easy instance of the 1in3SAT problem
    """
    _, hard, easy = Legacy_genHard1in3SATClausePair(nsize)
    rat = int(len(hard) / len(easy)) + 1
    return HardOneInThreeSAT(nsize, hard), HardOneInThreeSAT(nsize, easy * rat)


if __name__ == "__main__":
    # reference
    # hard1in3 = HardOneInThreeSAT(
    #     5,
    #     [[i, (i + 1) % 5, (i + 2) % 5] for i in range(5)] + [
    #         [0, 1, 2], [1, 0, 2], [2, 0, 1], [0, 2, 1], [0, 1, 3], [0, 3, 1], [1, 0, 3], [3, 0, 1],
    #         [0, 2, 3], [0, 3, 2], [2, 0, 3], [3, 0, 2]
    #     ] + [
    #         [4, 1, 2], [1, 4, 2], [2, 4, 1], [4, 2, 1], [4, 1, 3], [4, 3, 1], [1, 4, 3], [3, 4, 1],
    #         [4, 2, 3], [4, 3, 2], [2, 4, 3], [3, 4, 2]
    #     ]
    # )
    # easy1in3 = HardOneInThreeSAT(
    #     5,
    #     [[i, (i + 1) % 5, (i + 2) % 5] for i in range(5)] * 6
    # )
    # hrd, esy = gen1in3SATPair(5)
    # print(esy)
    # print(hrd)

    gtter = GSATN16C112Getter
    print(gtter[1])
    print(len(gtter))
    cls, confs = gtter[1]["clause"], gtter[1]["conf"]
    print(confs)
    exit(0)

    _, GSATN16C128HardestInfo, _, _ = GSATN16C128HardestGet()
    _, GSATN12C72HardClause = GSATN16C128ClausesPairGet()
    import json
    with open("./N16C128Clauses/hardest_mapping.json", "w") as f:
        json.dump(GSATN16C128HardestInfo, f, indent="\t")

    pr = HardGeneralSAT_normalized(12, GSATN12C72HardClause[92])
    size, spec = pr.info()
    print(pr.solution())
    print(pr.level_detail(1))
    print(pr.essential_clauses(1))
