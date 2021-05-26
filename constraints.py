# calculating and solving the constraint equations coming from the source - observer relation.
import numpy as np
import sys, os
from scipy import optimize as opt
import random
import matplotlib.pyplot as plt
import UserInput as UI

if not sys.version_info[0] < 3:
    from importlib import reload
reload(UI)
import globalvars as gv



# ------------------------------#unit vectors------------------------
xhat = np.array([1, 0, 0])
yhat = np.array([0, 1, 0])
zhat = np.array([0, 0, 1])


def frhat(t, p):  # theta and phi in radians   -- Spherical Radius
    return (
        np.sin(t) * np.cos(p) * xhat + np.sin(t) * np.sin(p) * yhat + np.cos(t) * zhat
    )


def fthat(t, p):  # theta and phi in radians  #latitude (azimuthal)
    return (
        np.cos(t) * np.cos(p) * xhat + np.cos(t) * np.sin(p) * yhat - np.sin(t) * zhat
    )


def fphat(t, p):  # theta and phi in radians  #longitude (polar)
    return -np.sin(p) * xhat + np.cos(p) * yhat


def frhohat(t, p):  # cylindrical radius
    return np.cos(p) * xhat + np.sin(p) * yhat


# ----------------------------- #Input Velocities -----
def fvi(t, p, delta):  # initial orientation of the TeV gamma ray
    return -np.cos(delta - t) * zhat + np.sin(delta - t) * frhohat(t, p)


def fvf(t, p):  # orientation of the GeV gamma ray
    return -frhat(t, p)


def fvavg(t, p, delta):
    return (-zhat) * np.cos(delta / 2.0 - t) + np.cross(fphat(t, p), zhat) * np.sin(
        delta / 2.0 - t
    )


# vavg here is vi+vf/||vi+vf||

# ----------------------------- #Define the possible BFields ------
def fB1(x, y, z):
    return -zhat * gv.B0


def fB2(x, y, z):
    return yhat * gv.B0


def fB3(x, y, z):
    return (
        np.cos(gv.beta) * np.cos(gv.beta2) * xhat
        + np.cos(gv.beta) * np.sin(gv.beta2) * yhat
        - np.sin(gv.beta) * zhat
    ) * gv.B0


def fB4(x, y, z):
    return (
        np.cos(2 * np.pi * z / gv.cl + gv.dalpha) * yhat
        + gv.hel * np.sin(2 * np.pi * z / gv.cl + gv.dalpha) * xhat
    ) * gv.B0


def fB5(x, y, z):
    return (
        np.cos(2 * np.pi * x / gv.cl + gv.dalpha) * yhat
        - gv.hel * np.sin(2 * np.pi * x / gv.cl + gv.dalpha) * zhat
    ) * gv.B0


if gv.case == 6:
    shat = np.array(
        [
            np.cos(gv.phi6) * np.sin(gv.theta6),
            np.sin(gv.theta6) * np.sin(gv.phi6),
            np.cos(gv.theta6),
        ]
    )
    temp = np.linalg.norm(np.cross(shat, xhat))
    if temp != 0:
        sp1 = -np.cross(shat, xhat) / temp
    else:
        sp1 = np.cross(shat, yhat) / np.linalg.norm(np.cross(shat, yhat))
    sp2 = np.cross(sp1, shat)


def fB6(x, y, z):
    s = x * shat[0] + y * shat[1] + z * shat[2]
    return (
        np.cos(2 * np.pi * s / gv.cl + gv.dalpha) * sp1
        + gv.hel * np.sin(2 * np.pi * s / gv.cl + gv.dalpha) * sp2
    ) * gv.B0


Bfname = UI.Bfieldname  #'sim_data/magfield_heleq1.npy'
Lfname = UI.Lcname  #'sim_data/Lcutoff_heleq1.npy'
# Lcutoff=np.load('sim_data/Lcutoff.npy')
# Blat=np.load('sim_data/magfield.npy')
if os.path.isfile(Bfname):
    Lcutoff = np.load(Lfname)
    Blat = np.load(Bfname)
else:
    Lcutoff = 10  # bogus
    Blat = np.zeros((3, 10, 10, 10))


def Binterp(x, y, z):
    nlat = len(Blat[0])
    n = np.array([x, y, z]) / Lcutoff + nlat / 2  # B array position of x,y,z
    B = np.zeros(3)
    if n.max() > nlat or n.min() < 0:
        print("A photon travelled beyond the Simulated Magnetic field's range")
        return B
    nfloor = np.floor(n)
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                w = (
                    3
                    - np.abs(n[0] - nfloor[0] - i)
                    - np.abs(n[1] - nfloor[1] - j)
                    - np.abs(n[2] - nfloor[2] - k)
                )
                B[0] = B[0] + w * Blat[0][nfloor[0] + i, nfloor[1] + j, nfloor[2] + k]
                B[1] = B[1] + w * Blat[1][nfloor[0] + i, nfloor[1] + j, nfloor[2] + k]
                B[2] = B[2] + w * Blat[2][nfloor[0] + i, nfloor[1] + j, nfloor[2] + k]
    return B


def fBfield(case):  # case 1 to case 5, This will have to be done in a better way.
    if case == 1:
        B = fB1
    if case == 2:
        B = fB2
    if case == 3:
        B = fB3
    if case == 4:
        B = fB4
    if case == 5:
        B = fB5
    if case == 6:
        B = fB6
    return B


# GENERATES THE MAGNETIC FIELD HERE
if gv.case == 0:
    if not os.path.isfile(Bfname):
        print("Problem: No Random Magnetic field B was supplied. B is set to 0")
    Bini = Binterp
    B = Binterp
else:
    Bini = fBfield(gv.case)
    B = fBfield(gv.case)


def fBrho(B, t, p):
    return np.dot(B, frhohat(t, p))


def fBphi(B, t, p):
    return np.dot(B, fphat(t, p))


def fBz(B, t, p):
    return np.dot(B, zhat)


def fvpara(B, t, p, delta):  # vi parallel to B
    return np.dot(B, fvi(t, p, delta))


def fvperp(vpara):  # vi normal to B
    return np.power(1.0 - vpara * vpara, 0.5)


def fdelx(
    vpara, v, vi, B, e, Egg
):  # time the electron moves (De/v), angular vel (v/R), speed of el (v,vi) , mag field B, charge e (+/- 1)
    # w2,tau2=w,tau
    if gv.STODE == 1:
        w2 = gv.drawMFPe(Egg, gv.zE(gv.dS)) / v
    else:
        w2 = gv.De(gv.zE(gv.dS), gv.Ee(Egg, gv.zE(gv.dS))) / v
    tau2 = v * 1.0 / gv.R(gv.Ee(Egg, gv.zE(gv.dS)), gv.B0, gv.zE(gv.dS))
    return v * (
        vpara * (tau2 - np.sin(w2 * tau2) / w2) * B
        + vi * np.sin(w2 * tau2) / w2
        + e * (np.cos(w2 * tau2) - 1.0) * np.cross(B, vi) / w2
    )


# --------- Constraints -------------------
def fLoS(t, delta, dg, dE):  # law of sines
    return np.sin(t) - dg * np.sin(delta) / dE


def fvcBdphi(bz, brho, delta, t):  # v cross B dot phi=0
    return np.sin(delta / 2.0 - t) * bz + np.cos(delta / 2.0 - t) * brho


def fsinphisol(B, t, delta):  # Solve for sin(phi) given the other variables.
    def f(p):
        return fvcBdphi(fBz(B, t, p), fBrho(B, t, p), delta, t)

    return np.sin(opt.newton(f, np.pi))


def fdeltaEqn(delta, vdB, Egg, wtau):
    tau2 = 1
    w2 = wtau
    return (1.0 - np.cos(delta)) - (1 - vdB * vdB) * (1 - np.cos(w2 * tau2))


def getxyz(dg, dE, angles):
    temp = dg * np.sin(angles[2] - angles[0])
    z = dE - dg * np.cos(angles[2] - angles[0])
    x = temp * np.cos(angles[1])
    y = temp * np.sin(angles[1])
    return x, y, z


def getxyzB(
    dg, dE, angles, fullsky=gv.fullsky
):  # gets the xyz coords from the blazar's frame of reference in order to determine the mag field position.
    if fullsky:
        temp = dg * np.sin(angles[2] - angles[0])
        vec = (
            gv.blazarloc
            - dg * np.cos(angles[2] - angles[0]) * gv.z2hat
            + temp * np.cos(angles[1]) * gv.x2hat
            + temp * np.sin(angles[1]) * gv.y2hat
        )
        x, y, z = vec[0], vec[1], vec[2]
    else:
        temp = dg * np.sin(angles[2] - angles[0])
        z = -dg * np.cos(angles[2] - angles[0])
        x = temp * np.cos(angles[1])
        y = temp * np.sin(angles[1])
    return x, y, z


def fcons1(dg, dE, Egg, angles):
    return fLoS(angles[0], angles[2], dg, dE)


def fcons2(dg, dE, Egg, angles):
    x, y, z = getxyz(dg, dE, angles)
    return fvcBdphi(
        fBz(B, angles[0], angles[1]),
        fBrho(B, angles[0], angles[1]),
        angles[2],
        angles[0],
    )


def fcons3(dg, dE, Egg, angles):
    x, y, z = getxyz(dg, dE, angles)
    return fdeltaEqn(angles[2], fvpara(B, angles[0], angles[1], angles[2]), Egg)


def fsolvecsq_dec2016(
    dg, dE, Egg, tG, pG, deltaG, ranget, Morp=False, xtol=1e-10, tau=False, RnB=False
):  # Solves the Constraints - a bit faster than the stable one and more likely to find a solution (I think).

    anglesG = np.array([tG, pG, deltaG])
    angles2G = np.array([pG, deltaG])

    x, y, z = getxyzB(dg, dE, anglesG)
    if not tau:
        if gv.STODE == 1 and not Morp:
            tau = gv.drawMFPe(Egg, gv.zE(gv.dS))

        else:
            tau = gv.De(gv.zE(gv.dS), gv.Ee(Egg, gv.zE(gv.dS)))
    w = 1.0 / gv.R(gv.Ee(Egg, gv.zE(gv.dS)), gv.B0, gv.zE(gv.dS))
    wtau = w * tau

    def F(angles2):
        angles = np.array(
            [np.arcsin(np.sin(angles2[1]) * dg / dE), angles2[0], angles2[1]]
        )
        x, y, z = getxyzB(dg, dE, angles)
        fB = B(x, y, z)
        bnorm = np.sqrt(fB[0] * fB[0] + fB[1] * fB[1] + fB[2] * fB[2])
        if RnB:
            w = bnorm / (RnB * 1e-14)
        else:
            w = 1.0 / gv.R(gv.Ee(Egg, gv.zE(gv.dS)), bnorm, gv.zE(gv.dS))
        wtau = w * tau
        # cons1=fLoS(angles[0],angles[2],dg,dE)
        cons2 = fvcBdphi(
            fBz(fB, angles[0], angles[1]),
            fBrho(fB, angles[0], angles[1]),
            angles[2],
            angles[0],
        )
        cons3 = fdeltaEqn(
            angles[2], fvpara(fB / bnorm, angles[0], angles[1], angles[2]), Egg, wtau
        )
        return (cons2, cons3)

    sol2, o1, ier, o2 = opt.fsolve(F, angles2G, xtol=xtol, full_output=1)

    sol = np.array([np.arcsin(np.sin(sol2[1]) * dg / dE), sol2[0], sol2[1]])

    charge = 1  # initialize the charge variable, now look what charge this has to be.
    if gv.chargeanalyze == 1:
        xvec = getxyz(dg, dE, sol)
        Bvec = B(xvec[0], xvec[1], xvec[2] + dE)
        orientation = np.dot(xvec, np.cross(fvi(sol[0], sol[1], sol[2]), Bvec))
        temp = np.dot(Bvec, fvi(sol[0], sol[1], sol[2])) / np.linalg.norm(Bvec)
        r = int(np.floor(wtau / (np.pi * np.power(1 - temp * temp, 0.5)))) % 2
        if orientation >= 0:  # the electron initially moves towards us
            if r == 0:  # electron must be the lepton
                charge = -1
            else:  # positron must be the lepton
                charge = 1
        if orientation < 0:  # the electron initially moves away from us
            if r == 0:  # positron must be the lepton
                charge = 1
            else:  # electron must be the lepton
                charge = -1
    return sol, ier, charge


def fsolvecsq_stable(
    dg, dE, Egg, tG, pG, deltaG, ranget, Morp=False, xtol=1e-10, tau=False, RnB=False
):  # Solves the Constraints - quite stable, but can jump to other branches
    anglesG = np.array([tG, pG, deltaG])

    x, y, z = getxyzB(dg, dE, anglesG)

    if not tau:
        if gv.STODE == 1 and not Morp:
            tau = gv.drawMFPe(Egg, gv.zE(gv.dS))
        else:
            tau = gv.De(gv.zE(gv.dS), gv.Ee(Egg, gv.zE(gv.dS)))
    w = 1.0 / gv.R(gv.Ee(Egg, gv.zE(gv.dS)), gv.B0, gv.zE(gv.dS))
    wtau = w * tau

    def F(angles):
        x, y, z = getxyzB(dg, dE, angles)
        fB = B(x, y, z)
        bnorm = np.sqrt(fB[0] * fB[0] + fB[1] * fB[1] + fB[2] * fB[2] + 1e-20)
        if RnB:
            w = bnorm / (RnB * 1e-14)
        else:
            w = 1.0 / gv.R(gv.Ee(Egg, gv.zE(gv.dS)), bnorm, gv.zE(gv.dS))
        wtau = w * tau
        cons1 = fLoS(angles[0], angles[2], dg, dE)
        cons2 = fvcBdphi(
            fBz(fB, angles[0], angles[1]),
            fBrho(fB, angles[0], angles[1]),
            angles[2],
            angles[0],
        )
        cons3 = fdeltaEqn(
            angles[2], fvpara(fB / bnorm, angles[0], angles[1], angles[2]), Egg, wtau
        )
        return (cons1, cons2, cons3)

    sol, o1, ier, o2 = opt.fsolve(F, anglesG, xtol=xtol, full_output=1)

    charge = 1  # initialize the charge variable, now look what charge this has to be.
    if gv.chargeanalyze == 1:
        xvec = getxyz(dg, dE, sol)
        Bvec = B(xvec[0], xvec[1], xvec[2] + dE)
        orientation = np.dot(xvec, np.cross(fvi(sol[0], sol[1], sol[2]), Bvec))
        temp = np.dot(Bvec, fvi(sol[0], sol[1], sol[2])) / np.linalg.norm(Bvec)
        r = int(np.floor(wtau / (np.pi * np.power(1 - temp * temp, 0.5)))) % 2
        if orientation >= 0:  # the electron initially moves towards us
            if r == 0:  # electron must be the lepton
                charge = -1
            else:  # positron must be the lepton
                charge = 1
        if orientation < 0:  # the electron initially moves away from us
            if r == 0:  # positron must be the lepton
                charge = 1
            else:  # electron must be the lepton
                charge = -1
    return sol, ier, charge


def fsolvecsq(
    dg, dE, Egg, tG, pG, deltaG, ranget, Morp=False, xtol=1e-10, tau=False, RnB=False
):
    return fsolvecsq_dec2016(
        dg, dE, Egg, tG, pG, deltaG, ranget, Morp=Morp, xtol=xtol, tau=tau, RnB=RnB
    )  # factor of 3 speed increase
    # return fsolvecsq_stable(dg,dE,Egg,tG,pG,deltaG,ranget,Morp=Morp,tau=tau,RnB=RnB)
