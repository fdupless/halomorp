import numpy as np
import sys, os

if not sys.version_info[0] < 3:
    from importlib import reload
from scipy import optimize as opt
from scipy import special as sp
from itertools import combinations
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import globalvars as gv


import stochMorp as sM

reload(sM)
import MCbackground as mcb

reload(mcb)


def setEbins(evtot):
    ebins = len(evtot)
    icount = 0
    maxEbin = True
    while maxEbin:
        if len(evtot[-icount - 1]) != 0:
            maxEbin = False
            usethis = ebins - 1 - icount
        icount = icount + 1
    return usethis, icount


import statisticfunctions as sf

reload(sf)


def calcqOriginal(e1vec, e2vec, e3vec):
    if len(e1vec) * len(e2vec) * len(e3vec) == 0:
        return [np.zeros(len(gv.region)), np.zeros(len(gv.region))]
    # matrix with all the scalar products of e3 and e1 or e2
    dote1e3 = np.dot(e3vec, e1vec.transpose())
    dote2e3 = np.dot(e3vec, e2vec.transpose())
    qval = np.zeros(len(gv.region))
    qerr = np.zeros(len(gv.region))
    for l in range(len(gv.region)):  # start from larger gv.region is faster
        rsize = np.cos(gv.region[len(gv.region) - 1 - l] * gv.degrad)
        # mask pairs that are too far away
        e1inregion = dote1e3 > rsize
        e2inregion = dote2e3 > rsize

        eta1 = np.zeros((e3vec.shape[0], 3))  # N_e3 values of eta1
        eta2 = np.zeros((e3vec.shape[0], 3))  # N_e3 values of eta2

        for t in range(len(eta1)):
            if e1vec[e1inregion[t]].shape[0] == 0 or e2vec[e2inregion[t]].shape[0] == 0:
                eta1[t] = np.zeros(3)
                eta2[t] = np.zeros(3)
            else:
                eta1[t] = np.average(e1vec[e1inregion[t]], axis=0)
                eta2[t] = np.average(e2vec[e2inregion[t]], axis=0)

            # calculate cross product
        eta1c2 = np.cross(eta1, eta2)  # this is an array of N_e3 vectors

        # dot with eta3 and sum, then divide by n3. store in qval array

        eta1c2d3 = np.diag(np.dot(e3vec, eta1c2.transpose()))
        qval[len(gv.region) - 1 - l] = eta1c2d3.mean()
        qerr[len(gv.region) - 1 - l] = eta1c2d3.std()

    # Divide the std in qerr by sqrt(N_e3)
    qerr /= np.sqrt(e3vec.shape[0])
    # Multiply everything by 10^6
    qval *= 1.0e6
    qerr *= 1.0e6
    return [qval, qerr]


def calcq(e1vec, e2vec, e3vec, CQ=0):
    if CQ == 0:
        cq = calcqOriginal(e1vec, e2vec, e3vec)
    if CQ == 1:
        cq = sf.calcqV3(e1vec, e2vec, e3vec)
    return cq


def genevents(
    ntsr,
    NumPhotons,
    ds=gv.dS,
    phi=0,
    omega=np.pi,
    alpha=0,
    jetact=0,
    SBL=np.array([0, 0]),
    RMF_K=np.array([0, 0, 0]),
):
    gv.jetactivated = 0
    if omega < np.pi:
        gv.jetactivated = 1
    if (
        SBL[0] != 0 or SBL[1] != 0
    ):  # SBL is the location of the blazar in radiants (b,l), (0,0) is the north pole
        gv.fullsky = True
        sM.setupB(SBL, ds)  # sets up the magnetic field in the correct orientation.
    if RMF_K[1] != 0:  # sets up a random magnetic field around the blazar
        sM.setnewB(RMF_K[0], int(RMF_K[1]), RMF_K[2])
    a1, a2, evtemp = sM.stomorp(
        ds, NumPhotons, alpha=alpha, phi=phi, omega=omega
    )  # events from the blazar
    evtemp = np.array(evtemp)
    EnBins = gv.EBins  # determining the Ebins.
    ev = [
        evtemp[(evtemp[:, 2] > EnBins[j]) & (evtemp[:, 2] < EnBins[j + 1])]
        for j in range(len(EnBins) - 1)
    ]
    ev = [
        ev[j][:, 0:2] for j in range(len(ev))
    ]  # events are binned per Energy and have the form [th,phi]
    # binning the blazar events
    maxangle = max(abs(ev[0][:, 0])) / gv.degrad
    gv.region = np.linspace(maxangle * 1.0 / gv.angnum, maxangle, num=gv.angnum)

    thcut = max([np.absolute(evtemp[j][0]) for j in range(len(evtemp))])
    thcut = max(thcut, 20 + 5)
    thcutHE = thcut - 20  # This cut is not right..
    evbacktemp = (
        np.array(mcb.SBbackground(NumPhotons, EnBins, ntsr, thcut, thcutHE)) * gv.degrad
    )  # Events of the background in radians
    # print('MC Events per Ebins ',[len(evbacktemp[i]) for i in range(len(evbacktemp))])
    evtot = np.array(
        [np.concatenate((ev[j], evbacktemp[j]), axis=0) for j in range(len(EnBins) - 1)]
    )
    evC = np.array(
        [
            np.array(
                [gv.angulartocart(ev[j][i][0], ev[j][i][1]) for i in range(len(ev[j]))]
            )
            for j in range(len(EnBins) - 1)
        ]
    )
    evbacktempC = np.array(
        [
            np.array(
                [
                    gv.angulartocart(evbacktemp[j][i][0], evbacktemp[j][i][1])
                    for i in range(len(evbacktemp[j]))
                ]
            )
            for j in range(len(EnBins) - 1)
        ]
    )
    evtotC = np.array(
        [
            np.array(
                [
                    gv.angulartocart(evtot[j][i][0], evtot[j][i][1])
                    for i in range(len(evtot[j]))
                ]
            )
            for j in range(len(EnBins) - 1)
        ]
    )

    return evtot, ev, evbacktemp, evtotC, evC, evbacktempC


def plotq(figname, Qtot, Qbla=False, Qmc=False):
    print("Generating the plot imgs/" + figname)
    combe12 = len(Qtot)
    fig = plt.figure(figsize=(6, 10))
    plt.title("orange=all, blue=MC, red=Blazar")
    plt.tick_params(labelbottom="off", labelleft="off")

    numba = 0
    for k in range(combe12):
        temp = fig.add_subplot(combe12, 1, k + 1)
        temp.plot(gv.region, Qtot[k, numba], marker="x", color="orange")
        temp.errorbar(gv.region, Qtot[k, numba], Qtot[k, numba + 1], color="orange")
        if Qmc:
            temp.plot(gv.region, Qmc[k, numba], marker=".", color="blue")
            temp.errorbar(gv.region, Qmc[k, numba], Qmc[k, numba + 1], color="blue")
        if Qbla:
            temp.plot(gv.region, Qbla[k, numba], marker="o", color="red")
            temp.errorbar(gv.region, Qbla[k, numba], Qbla[k, numba + 1], color="red")
        if k < combe12 - 1:
            temp.tick_params(axis="x", which="both", labelbottom="off", labelsize=5)
            temp.tick_params(axis="y", which="both", labelsize=10)
        else:
            temp.tick_params(which="both", labelsize=10)

    plt.ylabel("Value of Q")
    plt.xlabel("Radius of Region in Degrees")
    path = "imgs/"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig("imgs/" + figname)
    plt.close()
    return


def analyzeq(evtot, evBl=False, evMc=False, blazarloc=False, CQ=0):

    # if(setEbins(evtot)[1]==0): #function defined above, checks if there is an event in the largest bin.
    #    return Qtot,Qbla,Qmc #no high energy events --- I think this is a vestige?

    if blazarloc:
        nume12 = len(evtot)
        combe12 = int(
            np.math.factorial(nume12)
            / np.math.factorial(nume12 - 2)
            / np.math.factorial(2)
        )
        e12comb = list(combinations(range(nume12), 2))

        Qbla = np.zeros((combe12, 2, len(gv.region)))
        Qmc = np.zeros((combe12, 2, len(gv.region)))
        Qtot = np.zeros((combe12, 2, len(gv.region)))
        blazarLoc = np.array([[0, 0, 1]])

        for i in range(combe12):  # we use combinations to cycle the bins
            e12 = e12comb[i]
            gv.iE = e12[0]
            gv.jE = e12[1]
            Qtot[i, 0:2] = calcq(evtot[e12[0]], evtot[e12[1]], blazarLoc, CQ=CQ)
            if evBl:
                Qbla[i, 0:2] = calcq(evBl[e12[0]], evBl[e12[1]], blazarLoc, CQ=CQ)
            if evMc:
                Qmc[i, 0:2] = calcq(evMc[e12[0]], evMc[e12[1]], blazarLoc, CQ=CQ)
    else:
        nume12 = len(evtot) - 1
        combe12 = int(
            np.math.factorial(nume12)
            / np.math.factorial(nume12 - 2)
            / np.math.factorial(2)
        )
        e12comb = list(combinations(range(nume12), 2))

        Qbla = np.zeros((combe12, 2, len(gv.region)))
        Qmc = np.zeros((combe12, 2, len(gv.region)))
        Qtot = np.zeros((combe12, 2, len(gv.region)))
        for i in range(combe12):  # we use combinations to cycle the bins
            e12 = e12comb[i]
            gv.iE = e12[0]
            gv.jE = e12[1]
            Qtot[i, 0:2] = calcq(evtot[e12[0]], evtot[e12[1]], evtot[nume12], CQ=CQ)
            if evBl:
                Qbla[i, 0:2] = calcq(evBl[e12[0]], evBl[e12[1]], evBl[nume12], CQ=CQ)
            if evMc:
                Qmc[i, 0:2] = calcq(evMc[e12[0]], evMc[e12[1]], evMc[nume12], CQ=CQ)
    return Qtot, Qbla, Qmc


def qstats(numruns, ntsr, NumPhotons, totalonly=False, jetact=0):
    evtot, evBl, evBa, evtotC, evBlC, evBaC = genevents(ntsr, NumPhotons)
    nume12 = gv.qEbinmax - 1
    combe12 = int(
        np.math.factorial(nume12) / np.math.factorial(nume12 - 2) / np.math.factorial(2)
    )
    Qtot, Qbla, Qmc = (
        np.zeros((numruns, combe12, 2, len(gv.region))),
        np.zeros((numruns, combe12, 2, len(gv.region))),
        np.zeros((numruns, combe12, 2, len(gv.region))),
    )
    if totalonly:
        evBlC = False
        evBaC = False

    Qtot[0], Qbla[0], Qmc[0] = analyzeq(evtotC, evBl=evBlC, evMc=evBaC)
    for i in range(1, numruns):
        print("Run " + str(i) + " of " + str(numruns))
        evtot, evBl, evBa, evtotC, evBlC, evBaC = genevents(ntsr, NumPhotons)
        if totalonly:
            evBlC = False
            evBaC = False
        Qtot[i], Qbla[i], Qmc[i] = analyzeq(evtotC, evBl=evBlC, evMc=evBaC)
    QtotA, QblaA, QmcA = (
        np.zeros((combe12, 2, len(gv.region))),
        np.zeros((combe12, 2, len(gv.region))),
        np.zeros((combe12, 2, len(gv.region))),
    )

    print("Finding the average and standard deviation")
    for j in range(combe12):
        for p in range(len(gv.region)):
            for i in range(numruns):
                QtotA[j, 0, p] = QtotA[j, 0, p] + Qtot[i, j, 0, p]
                if not totalonly:
                    QblaA[j, 0, p] = QblaA[j, 0, p] + Qbla[i, j, 0, p]
                    QmcA[j, 0, p] = QmcA[j, 0, p] + Qmc[i, j, 0, p]
            QtotA[j, 0, p] = QtotA[j, 0, p] / numruns  # average
            if not totalonly:
                QblaA[j, 0, p] = QblaA[j, 0, p] / numruns
                QmcmcA[j, 0, p] = QmcmcA[j, 0, p] / numruns
            for i in range(numruns):
                QtotA[j, 1, p] = QtotA[j, 1, p] + np.power(
                    Qtot[i, j, 0, p] - QtotA[j, 0, p], 2.0
                )
                if not totalonly:
                    QblaA[j, 1, p] = QblaA[j, 1, p] + np.power(
                        Qbla[i, j, 0, p] - QblaA[j, 0, p], 2.0
                    )
                    QmcA[j, 1, p] = QmcA[j, 1, p] + np.power(
                        Qmc[i, j, 0, p] - QmcA[j, 0, p], 2.0
                    )
            QtotA[j, 1, p] = np.power(QtotA[j, 1, p] / (numruns - 1), 0.5)  # sample std
            if not totalonly:
                QblaA[j, 1, p] = np.power(QblaA[j, 1, p] / (numruns - 1), 0.5)
                QmcA[j, 1, p] = np.power(QmcA[j, 1, p] / (numruns - 1), 0.5)
    return QtotA, QblaA, QmcA
