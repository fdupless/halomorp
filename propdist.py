import time, sys, os
import numpy as np

if not sys.version_info[0] < 3:
    from importlib import reload
import globalvars as gv


import matplotlib.pyplot as plt


def Eg(E, z=0.24):
    # return 77.0*4*E*E/100.0*1.24/(1+z)
    return E * E / (gv.me * gv.me) * 1e6 * gv.ECMB(z) * 4.0 / 3.0


Ecutoff = gv.Ee(gv.Emin, gv.zE(gv.dS)) / 1000
Eemax = (gv.Esourcecutoff + 0.5) / 2.0
Eini = 5
Nsim = 10000
nbins = 100

fourpialphasquare = 4 * np.pi / (137.036 ** 2)
ncmb = 3.71 * 10 ** 8
mtokpc = 3.24e-20
TeVtokpc = 8.065e17 / mtokpc
meTeV = 0.511e-6
n0 = 3.71e8 / (mtokpc ** 3)

NUMde = TeVtokpc ** 2 / (n0 * fourpialphasquare)


def newE(E, z=0.24):
    return E - Eg(E, z=z) / 100.0


c = 299792458
sT = 6.65e-29
Urad = 0.25e-18


def f_lmfp(E, z=0.24):
    dEodt = 1.33 * sT * c * Urad * (E * E) / (meTeV ** 2)
    dE = E - newE(E)
    mfp = c * mtokpc * dE / dEodt
    return mfp


def de(E, z=0.24):
    # ncmb=n0*(1+z)**3
    # sigma_c=fourpialphasquare/(E*E*TeVtokpc**2)*np.log(E/meTeV)
    # lmfp=1/(ncmb*sigma_c)
    # lmfp=NUMde*E*E/(np.log(E/meTeV)*(1+z)**3)
    # lmfp=f_lmfp(E,z=0.24)
    lmfp = 1 / (sT * ncmb) * mtokpc
    return np.random.exponential(lmfp)


epsilonkek = 0.001


def gencascade(Eini=Eini, z=0.24):
    E = Eini
    pd = [[de(E, z=z), Eg(E, z=z) + epsilonkek * np.random.uniform()]]
    E = newE(E, z=z)
    while E > Ecutoff:
        pd.append(
            [de(E, z=z) + pd[-1][0], Eg(E, z=z) + epsilonkek * np.random.uniform()]
        )
        E = newE(E, z=z)
    return np.array(pd)


def RvsScat(Eini, z=0.24):
    R = [[0.52094 * (Eini / 5.0) * 1.0 / np.power(1.0 + z, 2.0), Eg(Eini)]]
    Ltraveled = f_lmfp(Eini, z=z)
    E = newE(Eini, z=z)
    while E > Ecutoff - 0.0001:
        Rnew = 0.52094 * (E / 5.0) * 1.0 / np.power(1.0 + z, 2.0)
        LTemp = Ltraveled
        Ltraveled += f_lmfp(E, z=z)
        R.append([(R[-1][0] * LTemp + Rnew * f_lmfp(E, z=z)) / Ltraveled, Eg(E)])
        E = newE(E, z=z)
    return np.array(R)


import scipy.stats as ss


def getdist(Eini=Eini, Nsim=Nsim, nbins=nbins, z=0.24):
    dist = []
    emax = 70
    Xmax = 1000
    raw = gencascade(Eini=Eini)
    for i in range(1, Nsim):
        raw = np.append(raw, gencascade(Eini=Eini, z=z), axis=0)

    # lengths=np.array(raw[:,0])
    # x=np.linspace(0,np.max(lengths),nbins)
    Energies = np.array(raw[:, 1])
    Eb = np.append([Eg(Ecutoff)], np.linspace(np.min(Energies) + 0.1, emax, nbins))
    aE, locE, scaleE = ss.gamma.fit(raw[:, 1], floc=Eg(Ecutoff - 0.01))
    distE = [aE, locE, scaleE]

    tmp = raw[np.where(np.logical_and(Energies > Eb[0], Energies < Eb[1]))]
    if len(tmp) > 0:
        a, loc, scale = ss.gamma.fit(tmp[:, 0], floc=0)
        dist.append([Eb[0], len(tmp), a, loc, scale, Xmax])
    else:
        dist.append([Eb[0], len(tmp), 1, 0, 0, Xmax])
    for i in range(1, nbins - 1):
        tmp = raw[np.where(np.logical_and(Energies > Eb[i], Energies < Eb[i + 1]))]
        if len(tmp) > 0:
            a, loc, scale = ss.gamma.fit(tmp[:, 0], floc=0)
            dist.append([Eb[i], len(tmp), a, loc, scale, Xmax])
        else:
            dist.append(
                [Eb[i], len(tmp), dist[-1][2], dist[-1][3], dist[-1][4], Xmax]
            )  # if no data, use the last bin's data
    dist = np.array(dist)
    dist[:, 1] /= np.max(dist[:, 1]) + epsilonkek * 1e-30

    R = RvsScat(Eini, z=0.24)
    return dist, distE, R


def LI(x1, x2, y1, y2, x):
    return (y2 - y1) * (x - x1) / (x2 - x1) + y1


def dlice(dist, pdf_E, dist2, pdf2, e0, e1, e2):

    if e0:
        pdf_En = [LI(e1, e2, pdf_E[i], pdf2[i], e0) for i in range(len(pdf_E))]
    logi = True
    Egg = 0
    icounter = 0
    while (Egg < gv.Emin or Egg > gv.Emax) and icounter < 100:
        icounter += 1
        Egg = ss.gamma.rvs(pdf_En[0], loc=pdf_En[1], scale=pdf_En[2])
        if icounter == 100:
            if Egg > gv.Emax:
                Egg = gv.Emax
            else:
                Egg = gv.Emin

    Eind = len(dist[:, 0][dist[:, 0] < Egg]) - 1
    Eind2 = len(dist2[:, 0][dist2[:, 0] < Egg]) - 1
    if Eind != len(dist) and e0:
        eg1, eg2 = dist[Eind, 0], dist[Eind + 1, 0]
        t2eg1, t2eg2 = dist[Eind, 0], dist[Eind + 1, 0]
        a1 = [
            LI(eg1, eg2, dist[Eind][2 + i], dist[Eind + 1][2 + i], Egg)
            for i in range(3)
        ]
        a2 = [
            LI(t2eg1, t2eg2, dist2[Eind][2 + i], dist2[Eind + 1][2 + i], Egg)
            for i in range(3)
        ]
        a = [LI(e1, e2, a1[i], a2[i], e0) for i in range(3)]
        traveldist = ss.gamma.rvs(a[0], loc=a[1], scale=a[2])
    else:
        traveldist = ss.gamma.rvs(dist[Eind, 2], loc=dist[Eind, 3], scale=dist[Eind, 4])

    return traveldist / 1000.0, Egg


def drawLeptonICevent(dist, pdf_E):
    logi = True
    Egg = 0
    icounter = 0
    while (Egg < gv.Emin or Egg > gv.Emax) and icounter < 100:
        icounter += 1
        Egg = ss.gamma.rvs(pdf_E[0], loc=pdf_E[1], scale=pdf_E[2])
        if icounter == 100:
            if Egg > gv.Emax:
                Egg = gv.Emax
            else:
                Egg = gv.Emin

    Eind = len(dist[:, 0][dist[:, 0] < Egg]) - 1
    # print(dist[Eind,2],dist[Eind,3],dist[Eind,4])
    traveldist = ss.gamma.rvs(dist[Eind, 2], loc=dist[Eind, 3], scale=dist[Eind, 4])
    return traveldist / 1000.0, Egg  # traveldistance in Mpc and Egg in GeV


Elist = [10, 5, 2]


def finddist(nbins):
    Em = np.max([Eemax, 10.0])
    Es = np.arange(Ecutoff, Em, Em / 100.0)

    path = "sim_data/dists/"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save("sim_data/dists/EbinList", Es)

    for i in range(len(Es)):
        print(i, len(Es))
        dist, Edist, R = getdist(Eini=Es[i], nbins=nbins)
        np.save("sim_data/dists/EbinList_" + str(i), dist)
        np.save("sim_data/dists/EbinListEdist_" + str(i), Edist)
        np.save("sim_data/dists/avgGyroRad_" + str(i), R)
    return


def plotdist(Dlist, xind=18, E=8):
    plt.figure()
    for d in Dlist:
        plt.plot(d[:, 0], d[:, 1])
    plt.title("PDF of Gamma ray emissions different locations")
    plt.xlabel("Distance (kpc)")
    plt.ylabel(r"$p_{\gamma}$ Prob density of $\gamma$ emission")
    plt.savefig("imgs/distribution.jpg")
    plt.clf()
    plt.close()
    plt.figure()
    for d in Dlist:
        g = lambda y: ss.gamma.pdf(y, d[xind, 5], loc=d[xind, 2], scale=d[xind, 3])
        x = np.linspace(d[xind, 2], d[xind, 4], 300)
        pdf = [g(y) for y in x]
        plt.plot(x, pdf)
    plt.title("PDF of Gamma ray emissions Energies at location d=18 kpc")
    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"$p_{\gamma}(E|x=x_0)$")
    plt.savefig("imgs/distributionenergy.jpg")
    plt.clf()
    plt.close()

    for d in Dlist:
        pdfxgE = []
        for xind in range(len(d)):
            g = lambda y: ss.gamma.pdf(y, d[xind, 5], loc=d[xind, 2], scale=d[xind, 3])
            pdfxgE.append(d[xind, 1] * g(E))
        plt.plot(d[:, 0], pdfxgE)
    plt.title("PDF of Gamma ray emissions of E=8 GeV at x")
    plt.xlabel("Distance (kpc)")
    plt.ylabel(r"$p_{\gamma}(x|E=E_0)$")
    plt.savefig("imgs/distributionenergyfixE.jpg")
    plt.clf()
    plt.close()

    return


def createdistest():
    dist = []
    for i in range(50000):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, test(1))
        if b < test(a):
            dist.append(a)
    plt.hist(dist, 50, normed=1, alpha=0.5)
    t = [test(x / 100.0) for x in range(100)]
    x = np.arange(100) / 100.0
    plt.plot(x, t)
    plt.show()
    return
