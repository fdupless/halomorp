#Using the functions developped in constraints to plot the Halo Morphology
import numpy as np
import sys
import matplotlib.pyplot as plt
import globalvars as gv
import constraints as co

#Set the magnetic field case in globalvars.py i.e. gv.case=#

#for case 1-2, the x,y,z of the magnetic field is not important. let them be zero.
#Elist=[10,13,15,18,20,23,25,30,40,100,150,200]


Emin=gv.Emin
Emax=gv.Emax


def findmorp(steps,ds,xtol=1e-10):
    solang=[]
    tcospsol=[[],[]]
    tsinpsol=[[],[]]
    tG,pG,deltaG=gv.tG,gv.pG,gv.deltaG
    for i in range(steps):
        Eexp=np.log10(Emin)+(np.log10(Emax)-np.log10(Emin))*np.power(1-i*1.0/(steps-1),2.0)
        Egg=np.power(10.0,Eexp)
        sol,ier,charge=co.fsolvecsq(gv.MFPg(gv.zE(ds),Egg),ds,Egg,tG,pG,deltaG,1,Morp=True,xtol=xtol)
        tG,pG,deltaG=sol[0],sol[1],sol[2]
        solang.append(sol)
        tcospsol[0].append(sol[0]*1.0/gv.degrad*np.cos(sol[1]))
        tsinpsol[0].append(sol[0]*1.0/gv.degrad*np.sin(sol[1]))

    tG,pG,deltaG=-gv.tG,gv.pG,-gv.deltaG
    for i in range(steps):
        Eexp=np.log10(Emin)+(np.log10(Emax)-np.log10(Emin))*np.power(1-i*1.0/(steps-1),2.0)
        Egg=np.power(10.0,Eexp)
        sol,ier,charge=co.fsolvecsq(gv.MFPg(gv.zE(ds),Egg),ds,Egg,tG,pG,deltaG,0,Morp=True)
        tG,pG,deltaG=sol[0],sol[1],sol[2]
        solang.append(sol)
        tcospsol[1].append(sol[0]*1.0/gv.degrad*np.cos(sol[1]))
        tsinpsol[1].append(sol[0]*1.0/gv.degrad*np.sin(sol[1]))
    return tcospsol,tsinpsol



def makemorpfig(fignameinput,tcospsol,tsinpsol,join):
    figname=fignameinput
    plt.figure()
    plt.scatter(tcospsol[0],tsinpsol[0],s=2)
    plt.scatter(tcospsol[1],tsinpsol[1],s=3)
    if(join==1):
        plt.plot(tcospsol[0],tsinpsol[0])
        plt.plot(tcospsol[1],tsinpsol[1])
    ylab='Transverse Extent '+r'$\theta$'+'sin('+r'$\phi$'+')'+' [deg]'
    plt.ylabel(ylab)
    xlab='Lateral Extent '+r'$\theta$'+'cos('+r'$\phi$'+')'+' [deg]'
    plt.xlabel(xlab)
    plt.savefig('imgs/'+figname)
    plt.close()
    return

