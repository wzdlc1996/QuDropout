import numpy as np
import problems

from intSampStat import superParams


import mkl
mkl.set_num_threads(1)

root_folder = superParams["root_folder"]
probl = problems.HardGeneralSAT_normalized(superParams["nsize"], superParams["cls"])
get_cumup = probl.cumu_prob_pair_getter()


def get_folder_name(prefix: str, dep: int, sampid: int) -> str:
    return f"{root_folder}/{prefix}" \
           f"/p={dep}/sample_%.3d/tracing_hist.npy" % sampid


def read_pval(prefix: str, dep: int, sampid: int) -> list:
    tracing = np.load(get_folder_name(prefix, dep, sampid), allow_pickle=True)
    return [x["prob"] for x in tracing]


def read_succp(prefix: str, dep: int, sampid: int) -> list:
    pval = read_pval(prefix, dep, sampid)
    return [np.array(get_cumup(x))[:, 1] for x in pval]


def read_last_gs(prefix: str, dep: int, sampid: int) -> float:
    tracing = np.load(get_folder_name(prefix, dep, sampid), allow_pickle=True)
    pval = tracing[-1]["prob"]
    return get_cumup(pval)[0][1]


def read_last_ener(prefix: str, dep: int, sampid: int) -> float:
    tracing = np.load(get_folder_name(prefix, dep, sampid), allow_pickle=True)
    pval = tracing[-1]["prob"]
    return float(probl.get_energy_loss(pval))


def save_gs():
    cate = ["RegQAOA", "DropRND", "DropHOMO"]
    for dep, samp in superParams["deps-sampl"].items():
        res = []
        ener = []
        for pref in cate:
            res.append([read_last_gs(pref, dep, n) for n in range(samp)])
            ener.append([read_last_ener(pref, dep, n) for n in range(samp)])

        np.savetxt(f"{root_folder}/p={dep}_samps.txt", res)
        np.savetxt(f"{root_folder}/p={dep}_ener.txt", ener)
    with open(f"{root_folder}/samps_info.txt", "w") as f:
        f.write("\t".join(cate))


if __name__ == "__main__":
    save_gs()
