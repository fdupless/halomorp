import numpy as np
import sys, os
from scipy import optimize as opt
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

if not sys.version_info[0] < 3:
    from importlib import reload
import randommagfield as rmf

reload(rmf)
import globalvars as gv

reload(gv)
import constraints as co

reload(co)
import morphology as morp

reload(morp)
import MCbackground as mcb

reload(mcb)


Emin = gv.Emin
Emax = gv.Emax
NumEventsDefault = 1000

path = "sim_data/dists/EbinList.npy"
if not os.path.exists(path):
    print("Distributions do not exist, setting UI.STODE==0")
    gv.STODE == 0
else:
    edists = np.load("sim_data/dists/EbinList.npy")
    dist = [
        np.load("sim_data/dists/EbinList_" + str(i) + ".npy")
        for i in range(len(edists))
    ]
    pdf_E = [
        np.load("sim_data/dists/EbinListEdist_" + str(i) + ".npy")
        for i in range(len(edists))
    ]
    RnoB = [
        np.load("sim_data/dists/avgGyroRad_" + str(i) + ".npy")
        for i in range(len(edists))
    ]
import propdist as pd

reload(pd)


def stomorp(ds, NumPhotons, alpha=0, phi=0, omega=np.pi, LeptonId=False, xtol=1e-10):
    B0 = gv.B0
    if LeptonId:
        gv.chargeanalyze = 1
    countphoton = 0
    events = []
    tcospsol = []
    tsinpsol = []
    tG, pG, deltaG = gv.tG, gv.pG, gv.deltaG
    solcount = 0
    while countphoton < NumPhotons:
        EggT = gv.drawEnergy()
        SignTheta = 0  # determining if the solution has theta positive or negative.
        tG, pG, deltaG = -gv.tG, gv.pG, -gv.deltaG
        # while(Egg<gv.Emin or Egg>gv.Emax):
        Eelini = gv.Ee(EggT, gv.zE(ds)) / 1000.0  # Ee initial in TeV
        while EggT < gv.Emin or Eelini * 2 > gv.Esourcecutoff:
            EggT = gv.drawEnergy()
            Eelini = gv.Ee(EggT, gv.zE(ds)) / 1000.0  # Ee initial in TeV
            # print(Egg,np.sqrt(Egg/77.0)*10)

        dg = gv.drawMFPg(EggT, gv.zE(ds))
        if gv.STODE == 1:
            ind = len(edists[edists < Eelini]) - 1
            tau0, Egg = pd.drawLeptonICevent(dist[ind], pdf_E[ind])
            # tau0=gv.drawMFPe(Egg,0.24)/10
            # if(ind!=len(edists)-1):
            # print(ind,Eelini,gv.Emin,EggT)
            # tau0,Egg=pd.dlice(dist[ind],pdf_E[ind],dist[ind+1],pdf_E[ind+1],Eelini,edists[ind],edists[ind+1])
            if len(RnoB[ind][RnoB[ind][:, 1] < Egg]) > 0:
                avgR = RnoB[ind][RnoB[ind][:, 1] < Egg][0, 0]
            else:
                avgR = RnoB[ind][-1, 0]
        else:
            Egg = EggT
            tau0 = gv.De(gv.zE(ds), gv.Ee(Egg, gv.zE(ds)))

        if np.random.uniform() > 0.5:
            SignTheta = 1
            tG, pG, deltaG = gv.tG, gv.pG, gv.deltaG
        # start=time.clock()
        sol, ier, charge = co.fsolvecsq(
            dg, ds, Egg, tG, pG, deltaG, SignTheta, xtol=tol, tau=tau0, RnB=avgR
        )
        # print(time.clock()-start)
        # if(ier!=1): print(solcount,countphoton)
        solcount = solcount + 1

        injet = True
        photonangle = sol[2] - sol[0]
        cosang = np.absolute(gv.angledotprod([photonangle, sol[1]], [alpha, phi]))
        if (
            gv.jetactivated == 1 and np.cos(omega) > cosang
        ):  # the second checks that the solution is part of the jet
            injet = False
            countphoton = countphoton + 1
        if (
            ier == 1 and np.absolute(sol[0]) < np.pi / 2.0 and injet
        ):  # ier=1 if a solution is found by fsolvecsq
            events.append([sol[0], sol[1]] + [Egg, charge])
            tcospsol.append(sol[0] * 1.0 / gv.degrad * np.cos(sol[1]))
            tsinpsol.append(sol[0] * 1.0 / gv.degrad * np.sin(sol[1]))
            countphoton = countphoton + 1
    return tcospsol, tsinpsol, events


def plotStoMorp(
    figname,
    phi=0,
    omega=np.pi,
    alpha=0,
    NumEvents=NumEventsDefault,
    Morp=False,
    LeptonId=False,
    blazloc=np.array([0, 0, 1]) * gv.dS,
    xtol=1e-10,
):
    res = 300
    gv.blazarloc = (
        blazloc
    )  # if fullsky is on, this allows us to check at blazar from different locations in a magnetic field.

    # import time
    # start=time.clock()
    tcospsol, tsinpsol, events = stomorp(
        gv.dS,
        NumEvents,
        alpha=alpha,
        phi=phi,
        omega=omega,
        LeptonId=LeptonId,
        xtol=1e-10,
    )
    # print(time.clock()-start)
    Ebins = gv.EBins
    events = np.array(events)
    # cols=[['red',0.3],['orange',0.4],['green',0.5],['blue',0.6],['purple',0.7]]

    hsv = plt.get_cmap("hsv")
    cNorm = colors.Normalize(vmin=0, vmax=len(gv.EBins) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hsv)  # color scheme

    Erange = [min(events[:, 2]), max(events[:, 2])]
    if Morp:
        tcospsolm, tsinpsolm = morp.findmorp(300, gv.dS)

    scaColorCP = [[] for i in range(len(Ebins) - 1)]
    scaColorSP = [[] for i in range(len(Ebins) - 1)]
    scaColorCE = [[] for i in range(len(Ebins) - 1)]
    scaColorSE = [[] for i in range(len(Ebins) - 1)]

    fig = plt.figure()
    gistrain = plt.get_cmap("gist_rainbow")
    eminColor = Erange[0]
    emaxColor = Erange[1]
    cNorm = colors.Normalize(vmin=eminColor, vmax=emaxColor)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=gistrain)  # color scheme

    if not LeptonId:  # color code the events according to the energy bins
        ax = fig.add_subplot(111)
        lines = []
        for i in range(len(events)):
            if np.floor(events[i][3]) == 1:
                ax.scatter(
                    tcospsol[i],
                    tsinpsol[i],
                    s=5,
                    marker="^",
                    color=scalarMap.to_rgba(events[i][2]),
                )
            else:
                ax.scatter(
                    tcospsol[i],
                    tsinpsol[i],
                    s=5,
                    marker="v",
                    color=scalarMap.to_rgba(events[i][2]),
                )

        ax.scatter([], [], s=5, marker="^", color="black", label="Positron")
        ax.scatter([], [], s=5, marker="v", color="black", label="Electron")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", fontsize="small")
        scalarMap.set_array(Erange)
        cbar = fig.colorbar(scalarMap)
        cbar.set_label("Photon Energy (GeV)", rotation=90, size=23)
        cbar.ax.tick_params(labelsize=20)
        ax.grid()
    else:  # Color code the events according to the Lepton (positron red, electron black)
        scaColorC = [[] for i in range(len(Ebins) - 1)]
        scaColorS = [[] for i in range(len(Ebins) - 1)]
        for i in range(len(events)):
            if np.floor(events[i][3]) == 1:
                scaColorC[0].append(tcospsol[i])
                scaColorS[0].append(tsinpsol[i])
            else:
                scaColorC[1].append(tcospsol[i])
                scaColorS[1].append(tsinpsol[i])
        ax = fig.add_subplot(111)
        lines = []
        retLine = ax.scatter(
            scaColorC[0], scaColorS[0], s=5, color="red", marker="^", label="positron"
        )
        retLine = ax.scatter(
            scaColorC[1], scaColorS[1], s=5, color="black", marker="v", label="electron"
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", fontsize="small")
        scalarMap.set_array(Erange)
        cbar = fig.colorbar(scalarMap)
        cbar.set_label("Photon Energy (GeV)", rotation=90, size=23)
        ax.grid()
    if Morp:
        plt.plot(tcospsolm[0], tsinpsolm[0], color="black", linewidth=1.0)
        plt.plot(tcospsolm[1], tsinpsolm[1], color="black", linewidth=1.0)
    # circsize=13
    # circ=plt.Circle((0,0),circsize,color='g',fill=False)
    # plt.gca().add_artist(circ)
    ylab = "Transverse Extent " + r"$\theta$" + "sin(" + r"$\phi$" + ")" + " [deg]"
    ax.set_ylabel(ylab, size=20)
    xlab = "Lateral Extent " + r"$\theta$" + "cos(" + r"$\phi$" + ")" + " [deg]"

    ax.set_xlabel(xlab, size=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.tight_layout()

    path = "imgs/"
    if not os.path.exists(path):
        os.makedirs(path)

    if gv.jetactivated == 0:
        plt.savefig("imgs/" + figname, dpi=res)
    else:
        plt.savefig(
            "imgs/" + figname, dpi=res
        )  # the line below makes figure with the parameters already written in the title
        # plt.savefig('imgs/'+figname+'_a'+str(int(np.floor(alpha/degrad)))+'_p'+str(int(np.floor(phi/degrad)))+'_o'+str(int(np.floor(omega/degrad))))

    plt.close()
    return


def plotevs(figname, ev, Morp=False, circ=False):
    tcospsol = np.array(
        [
            [
                ev[j][i][0] * np.cos(ev[j][i][1]) * 1 / gv.degrad
                for i in range(len(ev[j]))
            ]
            for j in range(len(ev))
        ]
    )
    tsinpsol = np.array(
        [
            [
                ev[j][i][0] * np.sin(ev[j][i][1]) * 1 / gv.degrad
                for i in range(len(ev[j]))
            ]
            for j in range(len(ev))
        ]
    )
    Ebins = gv.EBins
    hsv = plt.get_cmap("hsv")
    cNorm = colors.Normalize(vmin=0, vmax=len(Ebins) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hsv)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if Morp:
        tcospsolm, tsinpsolm = morp.findmorp(600, gv.dS)
        ax.scatter(tcospsolm[0], tsinpsolm[0], s=1.5)
        ax.scatter(tcospsolm[1], tsinpsolm[1], s=1.5)
        # plt.plot(tcospsolm[0],tsinpsolm[0],color='black',linewidth=1.0)
        # plt.plot(tcospsolm[1],tsinpsolm[1],color='black',linewidth=1.0)

    lines = []
    for i in range(len(Ebins) - 1):
        retLine = ax.scatter(
            tcospsol[i],
            tsinpsol[i],
            s=2,
            color=scalarMap.to_rgba(i),
            label="%.1f" % Ebins[i] + "-" + "%.1f" % Ebins[i + 1] + " GeV",
        )
    lines.append(retLine)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize="small")
    ax.grid()
    if circ:
        circsize = 25
        circ = plt.Circle((0, 0), circsize, color="g", fill=False)
        plt.gca().add_artist(circ)

    ylab = "Transverse Extent " + r"$\theta$" + "sin(" + r"$\phi$" + ")" + " [deg]"
    plt.ylabel(ylab)
    xlab = "Lateral Extent " + r"$\theta$" + "cos(" + r"$\phi$" + ")" + " [deg]"
    plt.xlabel(xlab)

    path = "imgs/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig("imgs/" + figname)
    plt.close()
    return


def blazarlocs(nevents, ns, bmin):  # we use galactic coordinates to generate blazars.
    zmin = np.sin(bmin * gv.degrad)  # latitude cut
    i = 0
    mcdirections = np.zeros((nevents, 2))
    while i < nevents:
        phi = np.random.uniform(0.0, 360.0)
        z = np.random.uniform(zmin, 1) * ns
        # sqz=np.sqrt(1-z**2)
        b = np.arcsin(z) / gv.degrad
        # mcdirections[i]=[sqz * np.cos(phi), sqz*np.sin(phi),z]
        mcdirections[i] = [90 - b, phi]
        i = i + 1
    return mcdirections


def transportcoordsys(angle):
    tz = co.zhat
    tpara = np.cos(angle[1]) * co.xhat + np.sin(angle[1]) * co.yhat
    tperp = np.sin(angle[1]) * co.xhat - np.cos(angle[1]) * co.yhat

    z2 = np.cos(angle[0]) * co.zhat
    y2 = (
        np.cos(angle[0]) * np.sin(angle[1]) * tpara
        + np.sin(angle[0]) * co.zhat
        - np.cos(angle[1]) * tperp
    )
    x2 = (
        np.cos(angle[0]) * np.cos(angle[1]) * tpara
        + np.sin(angle[0]) * co.zhat
        + np.sin(angle[1]) * tperp
    )

    return x2, y2, z2


def setupB(blazlocs, ds):
    gv.blazarloc = (
        np.array(
            [
                np.sin(blazlocs[0]) * np.cos(blazlocs[1]),
                np.sin(blazlocs[0]) * np.sin(blazlocs[1]),
                np.cos(blazlocs[0]),
            ]
        )
        * ds
    )
    gv.x2hat, gv.y2hat, gv.z2hat = transportcoordsys(blazlocs)

    def Bhalo(x, y, z):
        Bz = np.dot(gv.Bini(x, y, z), gv.z2hat)
        By = np.dot(gv.Bini(x, y, z), gv.y2hat)
        Bx = np.dot(gv.Bini(x, y, z), gv.x2hat)
        return np.array([Bx, By, Bz])

    gv.B = Bhalo
    return


def allskyblazarsevents(numblazars):
    blazlocs = np.concatenate(
        [blazarlocs(numblazars[0], 1.0, 0), blazarlocs(numblazars[1], -1.0, 0)]
    )
    gammaevents = []
    blazloc2 = []
    for j in range(len(blazlocs)):
        alpha = np.random.uniform(-1, 1) * 180 * gv.degrad
        omega = (20 + np.random.uniform() * 10) * gv.degrad
        phi = np.random.uniform() * 360 * gv.degrad
        ds = gv.drawSourceDist()
        blazloc2.append([blazlocs[j][0], blazlocs[j][1], ds])
        setupB(
            blazlocs[j], ds
        )  # sets up the magnetic field in the correct orientation.

        NumPhotons = gv.drawNumPhotons(ds, gv.flux, 1.0)
        tcospsol, tsinpsol, events = stomorp(
            ds, NumPhotons, alpha=alpha, phi=phi, omega=omega
        )  # create the events.
        for k in range(len(tcospsol)):
            b = (blazlocs[j][0] + tsinpsol[k]) % (360.0)
            l = (blazlocs[j][1] + tcospsol[k]) % (360.0)
            if b > 180:
                b = 180 - (b - 180)
                l = (l + 180) % 360
            b = 90 - b  # putting the events back to 90, -90.
            gammaevents.append([b, l, events[k][2]])
    return gammaevents, blazloc2


def morpofblazar(blazlocs2):
    dotsnum = 300
    blazmorpB = []
    blazmorpL = []
    for j in range(len(blazlocs2)):
        setupB([blazlocs2[j][0], blazlocs2[j][1]], blazlocs2[j][2])
        tc, ts = morp.findmorp(dotsnum, blazlocs2[j][2])
        for k in range(dotsnum):
            b = (blazlocs2[j][0] + ts[0][k]) % (360.0)
            l = (blazlocs2[j][1] + tc[0][k]) % (360.0)
            if b > 180:
                b = 180 - (b - 180)
                l = (l + 180) % 360
            blazmorpB.append(b)
            blazmorpL.append(l)
            b = (blazlocs2[j][0] + ts[1][k]) % (360.0)
            l = (blazlocs2[j][1] + tc[1][k]) % (360.0)
            if b > 180:
                b = 180 - (b - 180)
                l = (l + 180) % 360
            blazmorpB.append(b)
            blazmorpL.append(l)
    return blazmorpB, blazmorpL


def plotevents(ntsr, events, blazlocs, figname, Morp=False):
    from mpl_toolkits.basemap import Basemap
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    res = 300
    bgcolor = "white"
    Ebins = gv.EBins
    hsv = plt.get_cmap("hsv")
    cNorm = colors.Normalize(vmin=0, vmax=len(gv.EBins) - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hsv)  # color scheme

    scaColorB = [[] for i in range(len(Ebins) - 1)]
    scaColorL = [[] for i in range(len(Ebins) - 1)]
    for i in range(len(events)):
        for j in range(len(Ebins) - 2):
            if np.floor(events[i][2] / Ebins[j]) == 1:
                scaColorB[j].append(events[i][0])
                scaColorL[j].append(events[i][1])
        if np.floor(events[i][2] / Ebins[len(Ebins) - 2]) >= 1:
            scaColorB[len(Ebins) - 2].append(events[i][0])
            scaColorL[len(Ebins) - 2].append(events[i][1])

    ncolsev = [len(scaColorB[i]) for i in range(len(Ebins) - 1)]
    backB = [[[], []] for i in range(len(Ebins) - 1)]
    backL = [[[], []] for i in range(len(Ebins) - 1)]
    numtot = sum(ncolsev)
    eventsbackN = mcb.backgroundevents(1, 0, ntsr, ncolsev, numtot)
    eventsbackS = mcb.backgroundevents(-1, 0, ntsr, ncolsev, numtot)
    for i in range(len(Ebins) - 1):
        backB[i][0] = [eventsbackN[i][j][0] for j in range(len(eventsbackN[i]))]
        backB[i][1] = [eventsbackS[i][j][0] for j in range(len(eventsbackS[i]))]
        backL[i][0] = [eventsbackN[i][j][1] for j in range(len(eventsbackN[i]))]
        backL[i][1] = [eventsbackS[i][j][1] for j in range(len(eventsbackS[i]))]

    if Morp:
        blazmorpB, blazmorpL = morpofblazar(blazlocs)  # blazars morphology

    evsize = 0.1
    fig = plt.figure()
    plt.subplot("311", axisbg=bgcolor)
    plt.title("Photons are color coded by energy. Purple to Red for High to Low")
    m = Basemap(projection="moll", lon_0=180.0, lat_0=0)
    plt.xlabel("Blazar Events with morphology in grey")
    if Morp:
        x, y = m(blazmorpL, blazmorpB)
        m.scatter(x, y, marker="x", s=1e-4, color="grey")
    for j in range(len(Ebins) - 1):
        x, y = m(scaColorL[j], scaColorB[j])
        m.scatter(x, y, marker=".", s=evsize, color=scalarMap.to_rgba(j))

    plt.subplot("312", axisbg=bgcolor)
    m = Basemap(projection="moll", lon_0=180.0, lat_0=0)
    plt.xlabel("Background Events, Noise/Signal=" + str(ntsr))
    for j in range(len(Ebins) - 1):
        x, y = m(backL[j][0], backB[j][0])
        m.scatter(x, y, marker=".", s=evsize, color=scalarMap.to_rgba(j))
        x, y = m(backL[j][1], backB[j][1])
        m.scatter(x, y, marker=".", s=evsize, color=scalarMap.to_rgba(j))

    plt.subplot("313", axisbg=bgcolor)
    m = Basemap(projection="moll", lon_0=180.0, lat_0=0)
    plt.xlabel("Combined Map")
    for j in range(len(Ebins) - 1):
        x, y = m(scaColorL[j], scaColorB[j])
        m.scatter(x, y, marker=".", s=evsize, color=scalarMap.to_rgba(j))
        x, y = m(backL[j][0], backB[j][0])
        m.scatter(x, y, marker=".", s=evsize, color=scalarMap.to_rgba(j))
        x, y = m(backL[j][1], backB[j][1])
        m.scatter(x, y, marker=".", s=evsize, color=scalarMap.to_rgba(j))

    path = "imgs/"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("imgs/" + figname, dpi=res)
    plt.close()
    return


def setnewB(kmode, Nmodes, helicity, option=1, B0=gv.B0):
    if option == 2:
        rmf.setB = rmf.createB_one_K(kmode, Nmodes, hel=helicity, B0=gv.B0)
    else:
        rmf.setB = rmf.createB_one_K_uni(kmode, Nmodes, hel=helicity, B0=gv.B0)
    co.B = rmf.Banalytic
    return
