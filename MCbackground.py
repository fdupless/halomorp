import numpy as np
import sys
import scipy.integrate as integrate
import globalvars as gv


def numback(E1,E2,numtot,Emin,Emax): #based off of dN/dE from 1410.3696
    gam=2.30 
    Ecut=250.
    Escale=0.1
    Int=1.0*numtot/(integrate.quad(lambda x: np.power(x/Escale,1-gam)*np.exp(-x/Ecut),Emin,Emax)[0])
    num=Int*integrate.quad(lambda x: np.power(x/Escale,1-gam)*np.exp(-x/Ecut),E1,E2)[0]
    return int(np.floor(num))

def mcsample(nevents,ns,bmin,expdataE,pixdeg,sourcemask,phid,thd):
    # ns can be 1 or -1 depending on N or South
    # bmin : angle at which we cut off photons
    # expdataE to thd, exposure function variables. set to 0 for the moment
    zmin=np.sin(bmin*gv.degrad)
    i=0
    mcdirections=np.zeros((nevents,2))
    while i < nevents:
        # phi is uniformly distributed over 0, 360
        phi=np.random.uniform(0.,360.)
        #numphi=int(np.floor(phi/phid))
        z=np.random.uniform(zmin,1)*ns
        bgal=np.arcsin(z)/gv.degrad
        mcdirections[i]=[bgal,phi]
        i=i+1
    return mcdirections
'''#Exposure function --- Keep for later.
	if(ns==1):
            zMask=int(np.floor((bgal-bmin)/thd))
        else:
            zMask=int(np.floor((bgal+bmin)/thd+nth))
        
        if(sourcemask[(1+ns)/2][numphi][zMask]): #True if not masked  
            # calculate angular coordinates in degrees for pixel mapping
            #bgal=np.arctan2(z,sqz)/degrad #galactic latitude in degrees
            arraynumL=int(np.floor(-phi/pixdeg+len(expdata[0])*0.5))
            arraynumB=int(np.floor(bgal/pixdeg+len(expdata)*0.5)) #array number of each coordinates
            temp=np.random.uniform(0,1)
            if(temp<expdataE[arraynumB][arraynumL]):#modulate by the exposure
                sqz=np.sqrt(1-z**2)
                phi=phi*degrad
                mcdirections[i]=[sqz * np.cos(phi), sqz*np.sin(phi),z]
                i=i+1
'''
   # return mcdirections

def backgroundevents(ns,bmin,ntsr,signalevs,numtot): #uses MCsample and outputs an array of background events over allsky
    #signalevs=[signalevs[i]*ntsr for i in range(len(signalevs))] #signalevs has length of Ebins and contains the signal # of events
    num=[numback(gv.EBins[j],gv.EBins[j+1],numtot*ntsr,gv.Emin,gv.Emax) for j in range(len(gv.EBins)-1)]
    eventsback=[mcsample(num[i],ns,bmin,0,0,0,0,0) for i in range(len(signalevs))]
    return eventsback

def MCsampleSB(num,thcut):#MC sample for the single blazar case
    i=0
    mcdirections=np.zeros((num,2))
    while i < num:
        # phi is uniformly distributed over 0, 360
        phi=np.random.uniform(0.,360.)
        #numphi=int(np.floor(phi/phid))
        th=np.random.power(2)*thcut
        mcdirections[i]=[th,phi]
        i=i+1
    return mcdirections
def SBbackground(numtot,EnBins,ntsr,thcut,thcutHE): #single Blazar background
    #print(EnBins)
    num=[numback(EnBins[j],EnBins[j+1],numtot*ntsr,gv.Emin,gv.Emax) for j in range(len(EnBins)-1)]
    events=[MCsampleSB(num[j],thcut) for j in range(len(EnBins)-2)]#LE gamma bins
    events=events+[MCsampleSB(num[-1],thcutHE)]#HE gamma bin
    #print(min(events[4][:,0]), len(events[4]),thcutHE)
    #print(events[4][:,0])
    return events











