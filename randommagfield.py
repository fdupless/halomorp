import numpy as np
import sys,os
import random
if(not sys.version_info[0]<3):
    from importlib import reload
import scipy.fftpack as scidft
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import expm3, norm
import globalvars as gv
reload(gv)

#Lcutoff=10.0 #cutoff at this value in Mpc
#BoxSize=1280
#nlattice=BoxSize/Lcutoff #half of the size of mag field/cutoff distance, both in Mpc

def n0(k):
    if(k[0]==0 and k[1]!=0):
        n0=np.array([1,0,0])
    else:
        n0=np.array([0,1,0])
    return n0

def e1(kvec):
    temp=n0(kvec)
    if(temp[0]==1):
        e=np.array([0,-kvec[2],kvec[1]])/np.sqrt(kvec[2]*kvec[2]+kvec[1]*kvec[1])
    else:
        e=np.array([kvec[2],0,-kvec[0]])/np.sqrt(kvec[2]*kvec[2]+kvec[0]*kvec[0])
    return e

def e2(kvec):
    temp=n0(kvec)
    if(temp[0]==1):
        a=kvec[2]*kvec[2]+kvec[1]*kvec[1]
        e=-np.array([-a,kvec[1]*kvec[0],kvec[0]*kvec[2]])/np.sqrt(kvec[2]*kvec[2]*kvec[2]*kvec[2]+kvec[1]*kvec[1]*kvec[1]*kvec[1]+2*kvec[1]*kvec[2]*kvec[1]*kvec[2]+kvec[0]*kvec[2]*kvec[0]*kvec[2]+kvec[0]*kvec[1]*kvec[0]*kvec[1])
    else:
        a=kvec[2]*kvec[2]+kvec[0]*kvec[0]
        e=-np.array([kvec[1]*kvec[0],-a,kvec[1]*kvec[2]])/np.sqrt(kvec[2]*kvec[2]*kvec[2]*kvec[2]+kvec[0]*kvec[0]*kvec[0]*kvec[0]+kvec[1]*kvec[2]*kvec[1]*kvec[2]+2*kvec[0]*kvec[2]*kvec[0]*kvec[2]+kvec[0]*kvec[1]*kvec[0]*kvec[1])
    return e

def Kp(kvec,xvec):
    kdx=np.dot(kvec,xvec)
    real=(e1(kvec)*np.cos(kdx)-e2(kvec)*np.sin(kdx))*np.power(2,-1/2)
    im=(e1(kvec)*np.sin(kdx)+e2(kvec)*np.cos(kdx))*np.power(2,-1/2)
    return real,im

def Km(kvec,xvec):
    kdx=np.dot(kvec,xvec)
    real=(e1(kvec)*np.cos(kdx)+e2(kvec)*np.sin(kdx))*np.power(2,-1/2)
    im=(e1(kvec)*np.sin(kdx)-e2(kvec)*np.cos(kdx))*np.power(2,-1/2)
    return real,im

def Kpvec(kvec):
    real=e1(kvec)*np.power(2,-1/2)
    im=e2(kvec)*np.power(2,-1/2)
    return real,im

def Kmvec(kvec):
    real=e1(kvec)*np.power(2,-1/2)
    im=-e2(kvec)*np.power(2,-1/2)
    return real,im

def spectE(k):
    E=gv.normEB*k*k
    return E


twopicube=16*np.pi*np.pi*np.pi
def calcEBnorm(kvecs,hel):
    numModes=len(kvecs)
    normEB=0        
    nkvec=kvecs[0][0]*kvecs[0][0]+kvecs[0][1]*kvecs[0][1]+kvecs[0][2]*kvecs[0][2]
    for i in range(numModes):
        nkvec=kvecs[i][0]*kvecs[i][0]+kvecs[i][1]*kvecs[i][1]+kvecs[i][2]*kvecs[i][2]
        if(nkvec<39.47841760435/(gv.Lcutoff*gv.Lcutoff)):
            temp=39.47841760435*nkvec*nkvec
            normEB=normEB+temp
    #normEB=gv.B0/np.sqrt(normEB*len(kvecs))
    gv.normEB=gv.B0*gv.B0/(2*twopicube*nkvec)*len(kvecs)
    return 

def drawB(k,helicity=0):
    hel=helicity
    #knorm=np.linalg.norm(k)
    knorm=np.sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2])
    temp=2*np.pi#/(knorm*knorm*knorm)
    twopisquare=39.47841760435
    sigmap=np.sqrt(twopisquare*temp*(1+hel)*spectE(knorm))#*temp
    sigmam=np.sqrt(twopisquare*temp*(1-hel)*spectE(knorm))#*temp 
    phasep,phasem=np.random.uniform(0,2*np.pi), np.random.uniform(0,2*np.pi)
    #magBp,magBm=np.sqrt(twopisquare*temp*(1+hel)*np.abs(np.random.normal(0,sigmap))),np.sqrt(twopisquare*temp*(1-hel)*np.abs(np.random.normal(0,sigmam))) #equation (23) in 1607.00320
    
    if(sigmam==0):
        magBm=0
    else:
        magBm=np.abs(np.random.normal(0,sigmam))
    
    if(sigmap==0):
        magBp=0
    else:
        magBp=np.abs(np.random.normal(0,sigmap)) #equation (23) in 1607.00320 -- sort of.
        
        
    Breal=np.power(2.0,-0.5)*((magBp*np.cos(phasep)+magBm*np.cos(phasem))*e1(k)+(-magBp*np.sin(phasep)+magBm*np.sin(phasem))*e2(k))
    Bim=np.power(2.0,-0.5)*((magBp*np.sin(phasep)+magBm*np.sin(phasem))*e1(k)+(magBp*np.cos(phasep)-magBm*np.cos(phasem))*e2(k)) #equation (25) in 1607.00320
    return Breal+1j*Bim

def createB(nlattice,Lcutoff,hel=0):
    itok=2*np.pi/Lcutoff*(nlattice-1)/nlattice
    Bx=np.zeros((nlattice,nlattice,nlattice))+1j*np.zeros((nlattice,nlattice,nlattice))
    By=np.zeros((nlattice,nlattice,nlattice))+1j*np.zeros((nlattice,nlattice,nlattice))
    Bz=np.zeros((nlattice,nlattice,nlattice))+1j*np.zeros((nlattice,nlattice,nlattice))
    kvecs=[]
    #Setting the norm
    for i in range (nlattice):
        for j in range(nlattice):
            for k in range(nlattice):
                kvecs.append([i,j,k]*itok)
    calcEBnorm(kvecs,hel)
    
    for i in range(nlattice/2+1):
        #print(i)
        for j in range(nlattice):
            for k in range(nlattice):
                if(i==0 and j==0 and k ==0):
                    Bx[0,0,0]=0
                    By[0,0,0]=0
                    Bz[0,0,0]=0
                else:
                    kint=np.array([i,j,k])
                    kvec=kint*itok
                    B=(drawB(kvec,helicity=hel)+drawB(np.array([nlattice-i,j,k])*itok,helicity=hel))/2.0
                    Bx[i,j,k]=B[0]
                    Bx[-i,-j,-k]=np.conj(B[0])
                    By[i,j,k]=B[1]
                    By[-i,-j,-k]=np.conj(B[1])
                    Bz[i,j,k]=B[2]
                    Bz[-i,-j,-k]=np.conj(B[2])
    for i in [0,nlattice/2]:
        for j in [0,nlattice/2]:
            for k in [0,nlattice/2]:
                if(i==0 and j==0 and k ==0):
                    Bx[0,0,0]=0
                    By[0,0,0]=0
                    Bz[0,0,0]=0
                else:
                    kint=np.array([i,j,k])
                    kvec=kint*itok
                    temp=drawB(kvec,helicity=hel)
                    B=(temp+np.conj(temp))
                    Bx[i,j,k]=B[0]
                    Bx[-i,-j,-k]=np.conj(B[0])
                    By[i,j,k]=B[1]
                    By[-i,-j,-k]=np.conj(B[1])
                    Bz[i,j,k]=B[2]
                    Bz[-i,-j,-k]=np.conj(B[2])
                    
    #Bx=Bx*np.power(nlattice,3)
    pBx=scidft.ifftn(Bx)
    pBy=scidft.ifftn(By)
    pBz=scidft.ifftn(Bz)
    return np.array([pBx,pBy,pBz])

def savemagfield(ds,Lcutoff,helicity=0,info='_'):
    gv.Lcutoff=Lcutoff
    gv.BoxSize=ds
    nlattice=int(ds/Lcutoff)
    B=np.real(createB(nlattice,Lcutoff,hel=helicity))
    np.save('sim_data/Lcutoff_'+info,float(Lcutoff))
    np.save('sim_data/magfield_'+info,B)
    return

def createB_one_K(knorm,NumModes,hel=0,B0=gv.B0):
    gv.B0=B0
    th=np.zeros(NumModes)
    phi=np.zeros(NumModes)
    kvecs=[[0,0,0] for i in range(NumModes)]
    Bkx=[[0,0] for i in range(NumModes)]
    Bky=[[0,0] for i in range(NumModes)]
    Bkz=[[0,0] for i in range(NumModes)]
    #const=knorm*knorm/(8*np.pi*np.pi*np.pi)
    for i in range(NumModes):
        th[i]=np.arccos(np.random.uniform(-1,1))
        phi[i]=np.random.uniform()*2*np.pi
        kvecs[i]=np.array([np.sin(th[i])*np.cos(phi[i]),np.sin(th[i])*np.sin(phi[i]),np.cos(th[i])])*knorm
    
    calcEBnorm(kvecs,hel)
    
    for i in range(NumModes):
        temp=drawB(kvecs[i],helicity=hel)
        #Bkx[i]=[temp[0],np.conj(temp[0])]
        #Bky[i]=[temp[1],np.conj(temp[1])]
        #Bkz[i]=[temp[2],np.conj(temp[2])]
        Bkx[i]=[temp[0],0]
        Bky[i]=[temp[1],0]
        Bkz[i]=[temp[2],0]
        
    return Bkx,Bky,Bkz,kvecs,knorm
    
def M(axis, theta):
    return expm3(cross(eye(3), axis/norm(axis)*theta))
    
    
def M(axis, theta):#rotation matrix - supply unit axis and angle 
    return expm3(cross(eye(3), axis/norm(axis)*theta))
    
def createB_one_K_uni(knorm,NumModes,hel=0,B0=gv.B0):
    kvecs=[]
    gv.B0=B0
    '''
    from astropy.io import fits
    axis=np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)])
    axis=axis/np.sqrt(np.dot(axis,axis))
    rotangle=np.random.uniform(0,1)*np.pi
    rotMat=M(axis,rotangle)#random rotation matrix
    
    hpcoords=fits.open('healpix/pixel_coords_map_nested_galactic_res'+str(res)+'.fits')#obtaining the HEALPix coords
    '''
    #const=knorm*knorm/(8*np.pi*np.pi*np.pi)
    th=np.arccos(np.linspace(0,2,NumModes+2)-1)
    th=th[1:len(th)-1]
    phi=np.linspace(0,2,NumModes+1)[:-1]*np.pi
    vecangles=list(itertools.product(th,phi))
    vecangles.append([0,0])
    for angle in vecangles:
        kvecs.append(np.array([np.sin(angle[0])*np.cos(angle[1]),np.sin(angle[0])*np.sin(angle[1]),np.cos(angle[0])])*knorm) 
    Bkx=[[0,0] for i in range(len(kvecs))]
    Bky=[[0,0] for i in range(len(kvecs))]
    Bkz=[[0,0] for i in range(len(kvecs))]
    calcEBnorm(kvecs,hel)
    
    for i in range(NumModes*NumModes):
        temp=drawB(kvecs[i],helicity=hel)
        Bkx[i]=[temp[0],0]
        Bky[i]=[temp[1],0]
        Bkz[i]=[temp[2],0]
    return Bkx,Bky,Bkz,kvecs,knorm

setB=createB_one_K(1,1)
def Banalytic(x1,y1,z1):
    global setB
    x=np.array([x1,y1,z1])
    Bkx,Bky,Bkz,kvec,knorm=setB
    B=np.zeros(3)
    numK=len(kvec)
    for i in range(numK):
        xdotk=np.dot(x,kvec[i])
        B[0]=B[0]+np.real(Bkx[i][0])*np.cos(xdotk)-np.imag(Bkx[i][0])*np.sin(xdotk)
        B[1]=B[1]+np.real(Bky[i][0])*np.cos(xdotk)-np.imag(Bky[i][0])*np.sin(xdotk)
        B[2]=B[2]+np.real(Bkz[i][0])*np.cos(xdotk)-np.imag(Bkz[i][0])*np.sin(xdotk)
    B[0]=B[0]*4./numK
    B[1]=B[1]*4./numK
    B[2]=B[2]*4./numK
    return B


def viewB(name,Brms=gv.B0,knorm=1,Nmodes=1,nlattice=30,zslice=0,Bcomp=0):
    from mpl_toolkits.mplot3d import Axes3D
    from pylab import meshgrid,imshow,contour,clabel,colorbar,axis,title,show
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    global setB
    gv.B0=Brms
    setB=createB_one_K(knorm,Nmodes)
    B=Banalytic
    arB=[[B(i,j,zslice)[Bcomp] for i in range(nlattice)] for j in range(nlattice)]
    x=np.arange(nlattice)
    X,Y=meshgrid(x,x)
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    surf=ax.plot_surface(X,Y,arB,rstride=1,cstride=1,cmap=cm.RdBu,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.savefig('imgs/'+name)
    plt.clf()
    return


def curlB(B,x,y,z,epsilon=0.001):#curl of the vector field B at point x,y,z
    dxB=(B(x+epsilon,y,z)-B(x-epsilon,y,z))/epsilon
    dyB=(B(x,y+epsilon,z)-B(x,y-epsilon,z))/epsilon
    dzB=(B(x,y,z+epsilon)-B(x,y,z-epsilon))/epsilon
    cB=np.array([dyB[2]-dzB[1],dzB[0]-dxB[2],dxB[1]-dyB[0]]) 
    return cB

def BcBinjetLOS(B,jetdir,maxdir=400,step=10):
    BcB=np.zeros( np.int(np.ceil(maxdir/step)) )
    BcBRA=np.zeros( np.int(np.ceil(maxdir/step)) )
    for i in range(len(BcB)):
        x,y,z=jetdir[0]*step*(i+1),jetdir[1]*step*(i+1),jetdir[2]*step*(i+1)
        BcB[i]=np.dot(B(x,y,z),curlB(B,x,y,z))
        BcBRA[i]=np.sum(BcB)/(i+1)
    return BcB,BcBRA
