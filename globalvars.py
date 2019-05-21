# defining some global variables and functions that are used throughout the code.

import sys
import numpy as np
import UserInput as UI

if not sys.version_info[0] < 3:
    from importlib import reload
reload(UI)

degrad = np.pi / 180.0


# ------ Initial Program input -----
STODE = UI.STODE  # Stochasticity in the MFP of the electron (1=on, 0=off)
STOMFPg = UI.STOMFPg  # Stochasticity in the MFP of the initial photon (1=on, 0=off)
jetactivated = UI.jetactivated  # 1 if there is a jet, 0 otherwise
tG = 0.3
pG = 0.01
deltaG = tG
Emin = UI.Emin
Emax = UI.Emax
EBins = UI.EBins
Ebins = len(EBins) - 1


# ---User supplied global variables---
case = UI.case
B0 = UI.B0  # magnetic field strenght
k = UI.k  # set the kappa constant to unity
dalpha = UI.dalpha  # mag field shift
hel = (
    UI.hel
)  # helicty of the magnetic field for case 4,5 and 6. It takes the value +/- 1
beta = UI.beta  # angle of mag field in case 3 (uniform in a single direction)
beta2 = np.pi / 2  # azimuthal angle of case 3. Only changed in the code.
cl = UI.cl  # coherence length of the magnetic field in Mpc cases {4,5,6}
theta6 = UI.theta6  # angle of mag field in case 6
phi6 = UI.phi6  # angle of mag field in case 6
dS = UI.dS  # distance of the source from earth in Mpc
fullsky = UI.fullsky  # determines if we do the fullsky or not.
flux = UI.numflux
Esourcecutoff = UI.Esourcecutoff

# ----Implicit Global Variables ---
chargeanalyze = 0  # global variable to determine if we
blazarloc = (
    np.array([0, 0, 1]) * dS
)  # variable determining the location of the blazar when simulating multiple ones in the sky
x2hat, y2hat, z2hat = (
    np.zeros(3),
    np.zeros(3),
    np.array([0, 0, 1]),
)  # new coord system around blazar
qEbinmax = len(EBins) - 1
Lcutoff = 0.001  # random magnetic field cutoff initialization
BoxSize = 500  # random magnetic field Box Size initialization
normEB = [B0, B0]  # norm to get the magnetic field initialization
angstep = 1
angnum = UI.angnum
minangle = 0
maxangle = UI.maxangle
region = np.linspace(minangle, maxangle, angnum)
iE = 0
jE = 1  # variable determining the Ebins of calcq
AngCO = 80 * degrad  # Cutoff for calcq

# --------  Global Variables ------
eV = 1.0e-9
me = 0.511e-3
qem = np.power(1.0 / 137.035999, 0.5)


def ECMB(z):  # CMB photon average energy at redshift z in GeV
    return 6.0e-4 * eV * (1 + z)


def zE(dS):  # redshift of source at dS in Mpc
    return dS / 4140.0


def Ee(Egg, zgg):  # electron energy from emitted photon energy in GeV and redshift
    return me * np.power(0.75 * Egg / (ECMB(zgg)), 0.5)


def Eg0(Ee):  # parent photon energy
    return 2.0 * Ee


def MFPg(zE, Egg):  # MFP of TeV Photon in Mpc
    return k * 80 * 1.0e4 / Eg0(Ee(Egg, zE)) * 1.0 / (1.0 + zE)


def De(zg, Ee):  # electron cooling distance in Mpc at redshift zg
    return 0.06867 * 5000 / Ee * 1.0 / np.power(1.0 + zg, 4.0)


def R(Ee, B0, zg):  # gyroradius :: Bfield in Gauss, Ee in GeV, zg in no units
    return 0.52094 * (Ee / 5000) * (1.0e-14 / B0) * 1.0 / np.power(1.0 + zg, 2.0)


def RL(R, vdB):  # larmor radius
    return R * np.power(1 - vdB * vdB, 0.5)


def orbitnum(De, R):  # cooling distance over gyroradius radius
    return De / R


def thetaext(De, dS, R, MFPg):  # ang radius of halo for small bending approx
    return MFPg * De / (dS * R)


def Egcrit(B0, zE):  # no clue
    return 1


def thetamax(dS, B0):  # theta in degrees
    return 6.6 * degrad * 1.0e3 / dS * np.power(B0 / 5.5e-14, -0.5)


def drawMFPg(Egg, zgg):
    if (
        STOMFPg == 0
    ):  # these can be changed with gv.STOMFPg=whatever in other specific codes.
        dg = MFPg(zE(dS), Egg)
    else:
        dg = np.random.exponential(MFPg(zE(dS), Egg))
        while dg > MFPg(zE(dS), Emin) or dg < MFPg(zE(dS), Emax):
            dg = np.random.exponential(MFPg(zE(dS), Egg))  # initial photon distance
    return dg


def drawMFPe(Egg, zgg):
    if STODE == 0:
        de = De(zgg, Ee(Egg, zgg))
    else:
        de = np.random.exponential(De(zgg, Ee(Egg, zgg)))  # electron cooling distance
        while de < De(zgg, Ee(Emax, zgg)) or de > De(zgg, Ee(Emin, zgg)):
            de = np.random.exponential(De(zgg, Ee(Egg, zgg)))

    return de


def drawSourceDist():
    ds = dS + np.random.normal(0, dS / 5, size=None)  # mean at dS with std of dS/10
    return ds


def drawNumPhotons(ds, flux, exp):
    draw = int(np.floor(flux * np.power(dS / ds, 2.0)))  # higher for closer blazars
    return draw


from scipy.stats import pareto


def drawEnergy():
    a = 0.75
    draw = pareto(a).rvs() * Emin
    # draw=np.random.exponential(8.0)+Emin
    # draw=(np.random.beta(1.0,4)*16)+Emin
    return draw


def angledotprod(angle1, angle2):
    zz = np.cos(angle1[0]) * np.cos(angle2[0])
    yy = np.sin(angle1[1]) * np.sin(angle1[0]) * np.sin(angle2[1]) * np.sin(angle2[0])
    xx = np.cos(angle1[1]) * np.sin(angle1[0]) * np.cos(angle2[1]) * np.sin(angle2[0])
    return xx + yy + zz


def angulartocart(th, phi):
    t, p = th, phi
    if t < 0:
        t = -t
        p = p + np.pi
    z = np.cos(t)
    y = np.sin(t) * np.sin(p)
    x = np.sin(t) * np.cos(p)
    return [x, y, z]
