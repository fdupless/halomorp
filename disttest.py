import stochMorp as sM
import time, sys, os
import numpy as np
import EvSim as es
import UserInput as UI
import matplotlib.pyplot as plt

if not sys.version_info[0] < 3:
    from importlib import reload
import globalvars as gv

reload(UI)


EminTEV = 2 * gv.Ee(UI.Emin, 0.24)
from scipy.stats import pareto

dmax = 20000


def drawE2():
    a = 1.5
    draw = dmax * 2
    while draw > dmax:
        draw = pareto(a).rvs() * EminTEV
    return draw


def drawE3():
    a = 3
    draw = dmax * 2
    while draw > dmax:
        draw = pareto(a).rvs() * EminTEV
    return draw


def drawE1():
    draw = dmax * 2
    while draw > dmax:
        draw = 2 * gv.Ee(gv.drawEnergy(), 0.24)
    return draw


def compare(nsim, bins):
    tev = [drawE2() for i in range(nsim)]
    gev = [drawE1() for i in range(nsim)]
    bad = [drawE3() for i in range(nsim)]
    plt.hist(gev, bins, normed=1, alpha=0.3)
    plt.hist(tev, bins, normed=1, alpha=0.3)
    plt.hist(bad, bins, normed=1, alpha=0.3)
    plt.show()
    return
