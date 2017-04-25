
import sys,os
import numpy as np
if(not sys.version_info[0]<3):
    from importlib import reload #reload function is from a module for python 3 and higher
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as cb
from matplotlib import gridspec
import matplotlib.lines as mlines
import time
import shutil
import matplotlib.cm as cmx
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import UserInput as UI
reload(UI)
import randommagfield as rmf
reload(rmf)
import qstatistics as qs
reload(qs)
import globalvars as gv
reload(gv)
import constraints as co
reload(co)
import propdist as pd
reload(pd)


nphotons=1000

    
def fixed_aspect_ratio(ratio,ax):
    '''
    Set a fixed aspect ratio on matplotlib plots 
    regardless of axis units
    '''
    xvals,yvals = ax.get_xlim(),ax.get_ylim()

    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    ax.set_aspect(ratio*(xrange/yrange), adjustable='box')

def createplot3DEvs(cn,i,jet=False,sizemarker=5):
    res=100
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    evsangle=np.loadtxt('sim_data/3devents/case'+str(cn)+'/3deventsangle')
    Erange=[min(evsangle[:,2]),max(evsangle[:,2])]
    surfaceData=np.loadtxt('sim_data/3devents/case'+str(cn)+'/3devents')
    evsangle=evsangle[evsangle[:,2].argsort()]
    surfaceData=surfaceData[evsangle[:,2].argsort()]
                       
    gistrain=plt.get_cmap('gist_rainbow')
    eminColor=Erange[0]
    emaxColor=Erange[1]
    cNorm  = colors.Normalize(vmin=eminColor, vmax=emaxColor)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=gistrain) # color scheme
    ax.scatter(0,0,0,marker='*',c='Orange',lw=0,s=400)
    for j in range(len(evsangle)):
        ax.scatter(surfaceData[j][0],surfaceData[j][1],surfaceData[j][2],c=scalarMap.to_rgba(evsangle[j][2]),lw=0,s=sizemarker)
    if(jet==True):
        jetData=[np.loadtxt('sim_data/3devents/case'+str(cn)+'/jetcart_'+str(i+1)) for i in range(njets)]
        jetDataPlot=np.array([[ [jetData[i][j][0] for j in range(len(jetData[i]))] ,
                          [jetData[i][j][1] for j in range(len(jetData[i]))] ,
                          [jetData[i][j][2] for j in range(len(jetData[i]))] ,
                           ] for i in range(njets)])
        ax.scatter(jetDataPlot[i][0],jetDataPlot[i][1],jetDataPlot[i][2],c='black',alpha=0.3,s=sizemarker)
  
    xlab,ylab,zlab='x (Mpc)','y (Mpc)','z (Mpc)'
    sizetick=10
    sizelab=13
    ax.set_xlabel(xlab,size=sizelab) 
    ax.xaxis.set_tick_params(labelsize=sizetick) 
    ax.set_ylabel(ylab,size=sizelab) 
    ax.yaxis.set_tick_params(labelsize=sizetick) 
    ax.set_zlabel(zlab,size=sizelab) 
    ax.zaxis.set_tick_params(labelsize=sizetick) 
    
    scalarMap.set_array(Erange)
    cbar=fig.colorbar(scalarMap,fraction=0.046*0.5)
    cbar.set_label('Photon Energy (GeV)',rotation=90,size=sizelab)
    cbar.ax.tick_params(labelsize=sizetick) 
    fixed_aspect_ratio(1,ax)
    plt.tight_layout(pad=0.6)
    plt.savefig('sim_data/3devents/case'+str(cn)+'/3dimg_'+str(i+1),dpi=res)
    plt.close()
    return
    

path='sim_data/dists/EbinList.npy'
if(not os.path.exists(path)):
    print('Distributions do not exist, setting UI.STODE==0')
    gv.STODE==0
else:
    edists=np.load('sim_data/dists/EbinList.npy')
    dist=[np.load('sim_data/dists/EbinList_'+str(i)+'.npy') for i in range(len(edists))]
    pdf_E=[np.load('sim_data/dists/EbinListEdist_'+str(i)+'.npy') for i in range(len(edists))]
    RnoB=[np.load('sim_data/dists/avgGyroRad_'+str(i)+'.npy') for i in range(len(edists))]


def drawens(N):
    countphoton=0
    draw=[]
    while(countphoton<N):
        EggT=gv.drawEnergy()
        Eelini=gv.Ee(EggT,gv.zE(gv.dS))/1000.0 #Ee initial in TeV
        while(EggT<gv.Emin or Eelini*2>gv.Esourcecutoff):
            EggT=gv.drawEnergy()
            Eelini=gv.Ee(EggT,gv.zE(gv.dS))/1000.0 #Ee initial in TeV
            #print(Egg,np.sqrt(Egg/77.0)*10)
        dg=gv.drawMFPg(EggT,gv.zE(gv.dS)) 
        Egg=EggT
        tau0=0
        if(gv.STODE==1):
            ind=len(edists[edists<Eelini])-1
            tau0,Egg=pd.drawLeptonICevent(dist[ind],pdf_E[ind])
            if(tau0==0):
                print(edists[ind],Egg,pdf_E[ind], 'kek',countphoton,ind)
        draw.append([Eelini,Egg,tau0,dg]) 
        countphoton+=1
    return np.array(draw)



def SimEvent(ds,NumPhotons,cn=1,rmf_switch=False,B0=gv.B0):
    if(rmf_switch):
        kmode=0.05
        Nmodes=10
        helicity=0
        np.save('Btest',rmf.createB_one_K(kmode,Nmodes,hel=helicity))
        rmf.setB=np.load('Btest.npy')
        co.B=rmf.Banalytic
    countphoton=0
    tol=1e-12
    events=[]
    evsangle=[]
    tG,pG,deltaG=gv.tG,gv.pG,gv.deltaG
    while(countphoton<NumPhotons):
        EggT=gv.drawEnergy()
        SignTheta=0 #determining if the solution has theta positive or negative.
        tG,pG,deltaG=-gv.tG,gv.pG,-gv.deltaG
        #while(Egg<gv.Emin or Egg>gv.Emax):
        Eelini=gv.Ee(EggT,gv.zE(ds))/1000.0 #Ee initial in TeV
        while(EggT<gv.Emin or Eelini*2>gv.Esourcecutoff):
            EggT=gv.drawEnergy()
            Eelini=gv.Ee(EggT,gv.zE(ds))/1000.0 #Ee initial in TeV
            #print(Egg,np.sqrt(Egg/77.0)*10)

        dg=gv.drawMFPg(EggT,gv.zE(ds)) 
        if(gv.STODE==1):
            ind=len(edists[edists<Eelini])-1
            tau0,Egg=pd.drawLeptonICevent(dist[ind],pdf_E[ind])
            #tau0=gv.drawMFPe(Egg,0.24)/10
            #if(ind!=len(edists)-1):
                #print(ind,Eelini,gv.Emin,EggT)
                #tau0,Egg=pd.dlice(dist[ind],pdf_E[ind],dist[ind+1],pdf_E[ind+1],Eelini,edists[ind],edists[ind+1])
            if(len(RnoB[ind][RnoB[ind][:,1]<Egg])>0):
                avgR=RnoB[ind][RnoB[ind][:,1]<Egg][0,0]
            else:
                avgR=RnoB[ind][-1,0]
        else:
            Egg=EggT
            tau0=gv.De(gv.zE(ds),gv.Ee(Egg,gv.zE(ds)))

        

        if(np.random.uniform()>0.5): 
            SignTheta=1
            tG,pG,deltaG=gv.tG,gv.pG,gv.deltaG
        #start=time.clock()
        sol,ier,charge=co.fsolvecsq(dg,ds,Egg,tG,pG,deltaG,SignTheta,xtol=tol,tau=tau0,RnB=avgR)
        #print(time.clock()-start)
        if(ier==1 and np.absolute(sol[0])<np.pi/2.0):#ier=1 if a solution is found by fsolvecsq
            solcart=[dg*np.sin(sol[2]-sol[0])*np.cos(sol[1]),dg*np.sin(sol[2]-sol[0])*np.sin(sol[1]),-dg*np.cos(sol[2]-sol[0])]
            soleves=[sol[2]-sol[0],sol[1],Egg,sol[0]]
            events.append(solcart)
            evsangle.append(soleves)
            countphoton=countphoton+1
    
    
    path='sim_data/3devents/case'+str(cn)+'/'
    if(not os.path.exists(path)):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    np.savetxt('sim_data/3devents/case'+str(cn)+'/3devents',events)
    np.savetxt('sim_data/3devents/case'+str(cn)+'/3deventsangle',evsangle)
    
    return events,evsangle
    
    
def createplotJetSky(fig,ax,jetang,Erange,lsizerescale=1):
    gistrain=plt.get_cmap('gist_rainbow')
    #eminColor=min(jetang[:,2])
    #emaxColor=max(jetang[:,2])
    eminColor=Erange[0]
    emaxColor=Erange[1]
    cNorm  = colors.Normalize(vmin=eminColor, vmax=emaxColor)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=gistrain) # color scheme
    
    jetang=jetang[jetang[:,2].argsort()]

    for j in range(len(jetang)):
        CP=[jetang[j][3]*180.0/np.pi*np.cos(jetang[j][1])]
        SP=[jetang[j][3]*180.0/np.pi*np.sin(jetang[j][1])]
        col=jetang[j][2]
        ax.scatter(CP,SP,s=50,color=scalarMap.to_rgba(col))
    scalarMap.set_array(Erange)
    cbar=fig.colorbar(scalarMap)
    cbar.set_label('Photon Energy (GeV)',rotation=90,size=23*lsizerescale)
    cbar.ax.tick_params(labelsize=20*lsizerescale)
    ylab='Transverse Extent '+r'$\theta$'+'sin('+r'$\phi$'+')'+' [deg]'
    ax.set_ylabel(ylab,size=23*lsizerescale)
    xlab='Lateral Extent '+r'$\theta$'+'cos('+r'$\phi$'+')'+' [deg]'
    
    ax.set_xlabel(xlab,size=23*lsizerescale) 
    ax.xaxis.set_tick_params(labelsize=17*lsizerescale) 
    ax.yaxis.set_tick_params(labelsize=17*lsizerescale)

def cpJS(cn,jet=True,name='jetskyimg_1',lsizerescale=1):
    res=200
    evsangle=np.loadtxt('sim_data/3devents/case'+str(cn)+'/3deventsangle')
    Erange=[min(evsangle[:,2]),max(evsangle[:,2])]         
    fig=plt.figure()
    ax=fig.add_subplot(111)
    if(not jet):
        jetang=evsangle #use all events, we don't just want to plot the jet.
    else:
        jetang=np.loadtxt('sim_data/3devents/case'+str(cn)+'/jetang_1')   
    createplotJetSky(fig,ax,jetang,Erange,lsizerescale=lsizerescale)
            
    plt.tight_layout(pad=0.6)
    plt.savefig('sim_data/3devents/case'+str(cn)+'/'+name,dpi=res)
    plt.close()
    return

def JetEvs(NumJets=10,omega=5*gv.degrad,cn=1,doplots=False,jetfix=True):
    events=np.loadtxt('sim_data/3devents/case'+str(cn)+'/3devents')
    evsangle=np.loadtxt('sim_data/3devents/case'+str(cn)+'/3deventsangle')
    jetcount=0
    attempts=0
    res=200
    if(jetfix): NumJets=1
    while(jetcount<NumJets):
        if(jetfix):
            jth=omega*np.random.uniform()
            jph=2*np.pi*np.random.uniform()
            #jth=5*gv.degrad
            #jph=2*np.pi*gv.degrad
            jetSel=[jth,jph]
            jetSelCart=[np.sin(jth)*np.cos(jph+np.pi),np.sin(jth)*np.sin(jph+np.pi),np.cos(jth)]
        else:
            eventSel=int(np.floor(np.random.uniform()*len(events)))
            jetSel=[evsangle[eventSel][0],evsangle[eventSel][1]]
            jetSelCart=events[eventSel]

        injet=[]
        for i in range(len(evsangle)):
            cosang=np.absolute(gv.angledotprod(jetSel,[evsangle[i][0],evsangle[i][1]]))
            if(np.cos(omega)>cosang):#the second checks that the solution is part of the jet
                injet.append(False)
            else:
                injet.append(True)
        injet=np.array(injet)
        jetevs=events[injet]
        Erange=[min(evsangle[:,2]),max(evsangle[:,2])]
        #if(len(jetevs)<=2): print('Not enough events')
        attempts=attempts+1
        if(len(jetevs)>2 or attempts==20):
            jetcount=jetcount+1
            attempts=0
            jetang=evsangle[injet]
            
            fig=plt.figure()
            ax=fig.add_subplot(111)
            createplotJetSky(fig,ax,jetang,Erange)
            
            plt.tight_layout(pad=0.6)
            plt.savefig('sim_data/3devents/case'+str(cn)+'/jetskyimg_'+str(jetcount),dpi=res)
            plt.close()
            
            np.savetxt('sim_data/3devents/case'+str(cn)+'/jetopenning_'+str(jetcount),[omega])
            np.savetxt('sim_data/3devents/case'+str(cn)+'/jetdir_'+str(jetcount),jetSelCart)
            np.savetxt('sim_data/3devents/case'+str(cn)+'/jetcart_'+str(jetcount),jetevs)
            np.savetxt('sim_data/3devents/case'+str(cn)+'/jetang_'+str(jetcount),jetang)
   
    if(doplots):          
        for i in range(NumJets):
            createplot3DEvs(cn,i)
    
    return 

def QJets(NumJets=1,cn=1,NumEBins=3,UseUIBins=False,Qstart='Q_',CQ=0):
    Q=[]
    evs=[[] for j in range(NumJets)]
    for jetcount in range(NumJets):
        evsangtemp=np.loadtxt('sim_data/3devents/case'+str(cn)+'/jetang_'+str(jetcount+1))
        evstemp=np.array([gv.angulartocart(evsangtemp[i][3],evsangtemp[i][1]) for i in range(len(evsangtemp))])

        En=evsangtemp[:,2]
        EnS=np.sort(En)
        ind=np.array([np.where(En==EnS[i])[0][0] for i in range(len(En))])
        if(UseUIBins):
            Ebins=UI.EBins
            for nbins in range(len(Ebins)-1):
                evs[jetcount].append(evstemp[np.logical_and(En>Ebins[nbins],En<Ebins[nbins+1])])
            np.savetxt('sim_data/3devents/case'+str(cn)+'/BinsSelection',[0])
        else:
            bins=int(np.floor(len(evstemp)/NumEBins))
            extra=len(evstemp)%NumEBins
            evs[jetcount].append(evstemp[ind[0:extra+bins]])
            Ebins=[1]
            for j in range(1,NumEBins):
                evs[jetcount].append(evstemp[ind[extra+bins*j:extra+bins*(j+1)]])
                Ebins.append(j+1)
            Ebins.append(NumEBins)
            np.savetxt('sim_data/3devents/case'+str(cn)+'/BinsSelection',[1])
        evs[jetcount]=np.array(evs[jetcount])
        Q.append(qs.analyzeq(evs[jetcount],blazarloc=True,CQ=CQ)[0]) #[0] chooses Qtot (analyzeq returns 3 different Q's)
        
        for p in range(len(Q[jetcount])):
            np.savetxt('sim_data/3devents/case'+str(cn)+'/'+Qstart+str(jetcount+1)+'_bin_'+str(p+1),Q[jetcount][p][0]) #save the mean Q
        np.savetxt('sim_data/3devents/case'+str(cn)+'/region_'+str(jetcount+1),gv.region)
    np.savetxt('sim_data/3devents/case'+str(cn)+'/UsedEbins',Ebins)
    return Q

def QJetsStatistics(NumJets=10,cn=1):
    Qjets=[]
    Qregion=[]
    region=np.loadtxt(directory+'region_1')
    angnum=len(region)

    combe12=len( [filename for filename in os.listdir('.') if filename.startswith('sim_data/3devents/case'+str(cn)+'/Q_'+str(1))])    #this might not work
    Qavgjets=[[[0 for i in range(angnum)] for j in range(combe12)],[[0 for i in range(angnum)] for j in range(combe12)]]
    directory='sim_data/3devents/case'+str(cn)+'/'

    for jetcount in range(NumJets):
        prefixed = [filename for filename in os.listdir(directory) if filename.startswith('Q_'+str(jetcount+1))]
        Qjets.append([np.loadtxt(directory+fn) for fn in prefixed])
        for binN in range(len(Qjets[jetcount])):
            for angN in range(len(Qjets[jetcount][binN])):
                Qavgjets[0][binN][angN]=Qavgjets[0][binN][angN]+Qjets[jetcount][binN][0][angN]
    for binN in range(len(Qavgjets[0])):
        for angN in range(len(Qavgjets[0][binN])):
            Qavgjets[0][binN][angN]=Qavgjets[0][binN][angN]/(1.0*NumJets)

    for jetcount in range(NumJets):
        for binN in range(len(Qavgjets[0])):
            for angN in range(len(Qavgjets[0][binN])):
                Qavgjets[1][binN][angN]=Qavgjets[1][binN][angN]+(Qjets[jetcount][binN][0][angN]-Qavgjets[0][binN][angN])*(Qjets[jetcount][binN][0][angN]-Qavgjets[0][binN][angN])
 
    for binN in range(len(Qavgjets[0])):
        for angN in range(len(Qavgjets[0][binN])):
            Qavgjets[1][binN][angN]=np.sqrt(Qavgjets[1][binN][angN])/(1.0*(NumJets-1))
    
    fig=plt.figure()
    plt.plot(region,Qavgjets[0],marker='o',color='blue')
    plt.errorbar(region,Qavgjets[0],Qavgjets[1],color='blue')
    plt.ylabel('Value of Q')
    plt.xlabel('Radius of Region in Degrees')
    fig.savefig('sim_data/3devents/case'+str(cn)+'/Qtotal',dpi=150)
    plt.close()    
    return Qavgjets


kmode=0.015
Nmodes=5
helicity=-1
rmf.setB=rmf.createB_one_K(kmode,Nmodes,hel=helicity)
co.B=rmf.Banalytic
njets=1
Nphotons=1000

def fcalcBcB(cns,i):
    directory='sim_data/3devents/case'+str(cns+i)+'/'
    jetdir=np.loadtxt(directory+'jetdir_1')
    step=min(10,1/kmode)
    BcB,BcBRA=rmf.BcBinjetLOS(co.B,jetdir,maxdir=400,step=step)
    np.savetxt(directory+'/BcB',BcB)
    np.savetxt(directory+'/BcBRA',BcBRA)
   
    LoSdist=np.arange(1,len(BcB)+1)*step
    
    labSize=17
    fig=plt.figure()
    plt.title('Physical Helicity along Jet',fontsize=labSize+2)
    plt.plot(LoSdist,BcB,color='blue',lw=2)
    #plt.plot(LoSdist,BcBRA,color='red',lw=2)
    plt.ylabel(r'$B \cdot \nabla\times B$ (G$^2$/Mpc)',fontsize=labSize+2)
    plt.xlabel('Distance from Blazar (Mpc)',fontsize=labSize+2)
    plt.xticks(size=labSize)
    plt.yticks(size=labSize)
    fig.savefig(directory+'/BcB',dpi=150)
    plt.close()    
    return    

#es.SimNHalos(N=382,Nmodes=5,hel=1,njets=1,Nphotons=1000,B0=5e-15,jetfix=True,cns=24618,kmode=1,CQ=1)

#es.SimNHalos(N=1000,Nmodes=5,hel=-1,njets=1,Nphotons=1000,B0=2e-15,jetfix=True,cns=49000,kmode=0.01,CQ=1)


def SimNHalos(N=2,kmode=kmode,Nmodes=Nmodes,hel=helicity,cns=100,njets=njets,
              Nphotons=Nphotons,MCB=False,B0=gv.B0,UseUIBins=True,case=False,
              jetfix=True,CalcBcB=False,CQ=0,omega=5*gv.degrad):

    gv.B0=B0
    for i in range(N):
        if(case): #chooses the magnetic field. If case=number, set the field to be that case. Otherwise we use mag field created from Monte Carlo draws or a "uniform"(labeled uni) one.
            co.B=co.fBfield(case)
        else:
            if(MCB):
                rmf.setB=rmf.createB_one_K(kmode,Nmodes,hel=hel,B0=B0)
            else:
                rmf.setB=rmf.createB_one_K_uni(kmode,Nmodes,hel=hel,B0=B0)
            co.B=rmf.Banalytic
        #start=time.clock()
        SimEvent(1000,Nphotons,cn=cns+i,B0=B0)
        #print(time.clock()-start)
        JetEvs(NumJets=njets,omega=omega,cn=cns+i,jetfix=jetfix)
        QJets(NumJets=njets,cn=cns+i,UseUIBins=UseUIBins,CQ=CQ)

        if(CalcBcB): fcalcBcB(cns,i)
        

    return

#em1=np.arange(1000)+7000
#e=[np.arange(1000)+8000,np.arange(1000)+9000]
def QsPlot(N=2,cns=100,name='_',njets=njets,plotallQ=True,EbinsMax=100,plotinCN=False,
    Qstart='Q_',EvsList=False,labelEbins=True,evsSP=False,screwupfactor=1,evstype='hel'):
    ThreeHels=type(evsSP)!=type(False)#do np.shape array, if false. then evsSP must be plotted over the same graph.
    s=screwupfactor#coord system handness screw up factor. (1 -> -1 to change the handedness)

    if(type(EvsList)==type(False)):
        evsnum=[]
        for i in range(N):
            evsnum.append(cns+i)
    else:
        evsnum=EvsList
    directory='sim_data/3devents/case'+str(evsnum[0])+'/'

    combe12=len([filename for filename in os.listdir(directory) if filename.startswith(Qstart+'1_bin')])
    
    region=np.loadtxt(directory+'region_1')
    angnum=len(region)

    Q=[[] for j in range(combe12)]
    
    Qtot=[[np.zeros(angnum),np.zeros(angnum)] for j in range(combe12)]

    linew=2 #linewidth of the plots
    if(ThreeHels): #do np.shape array, if false. then evsSP must be plotted over the same graph.
        linew=1 #linewidth of the plots
        plotallQ=False
        Qextra=[[[] for j in range(combe12)],[[] for j in range(combe12)]]
        Qextratot=[[[np.zeros(angnum),np.zeros(angnum)] for j in range(combe12)],[[np.zeros(angnum),np.zeros(angnum)] for j in range(combe12)]]
        for INT in range(2):
            for i in evsSP[INT]:
                directory='sim_data/3devents/case'+str(i)+'/'
                Qfiles= [[directory+filename for filename in os.listdir(directory) if np.logical_and(filename.startswith(Qstart),filename.endswith('bin_'+str(k+1)))] for k in range(combe12)] #Qfiles[i] contains the Q values for the Ebin i+1 for all the jets.
                for j in range(combe12):
                    for qfn in Qfiles[j]:
                        temp=np.loadtxt(qfn)
                        if(not np.array_equal(temp,Qtot[0][0])):#averaging over the Q's that deviates from 0 only.
                            Qextra[INT][j].append(temp)
                for k in range(len(Qextra[INT])):# loop over bins
                    if(len(Qextra[INT][k])>0):
                        temp=np.array([Qextra[INT][k][i] for i in range(len(Qextra[INT][k]))])
                    else:
                        temp=np.array([np.zeros(angnum)])
                    Qextratot[INT][k][0]=np.mean(temp,axis=0)
                    if(len(temp)<=1):
                        Qextratot[INT][k][1]=0
                    else:
                        Qextratot[INT][k][1]=np.std(temp,axis=0)/np.sqrt(len(temp)-1)
        #Qextratot=-np.array(Qextratot)
    for i in evsnum:
        directory='sim_data/3devents/case'+str(i)+'/'
        Qfiles= [[directory+filename for filename in os.listdir(directory) if np.logical_and(filename.startswith(Qstart),filename.endswith('bin_'+str(k+1)))] for k in range(combe12)] #Qfiles[i] contains the Q values for the Ebin i+1 for all the jets.
        for j in range(combe12):
            for qfn in Qfiles[j]:
                temp=np.loadtxt(qfn)
                if(not np.array_equal(temp,Qtot[0][0])):#averaging over the Q's that deviates from 0 only.
                    Q[j].append(temp)
    
    Ebins=np.loadtxt('sim_data/3devents/case'+str(evsnum[0])+'/UsedEbins')
    e12comb=list(combinations(range(len(Ebins)-1),2))

    if(np.loadtxt('sim_data/3devents/case'+str(evsnum[0])+'/BinsSelection')==0):#This uses UI bins
        plotlabels=[str(Ebins[e12comb[iloop][0]])+'-'+str(Ebins[e12comb[iloop][0]+1])+' GeV \n'+str(Ebins[e12comb[iloop][1]])+'-'+str(Ebins[e12comb[iloop][1]+1])+' GeV'  for iloop in range(combe12)]
    else: #Likely use AutoBins --- Unless more options were added.
        plotlabels=['Bins '+str(int(Ebins[e12comb[iloop][0]]))+'&'+str(int(Ebins[e12comb[iloop][1]]))  for iloop in range(combe12)]
    
    for k in range(len(Q)):# loop over bins
        if(len(Q[k])>0):
            temp=np.array([Q[k][i] for i in range(len(Q[k]))])
        else:
            temp=np.array([np.zeros(angnum)])
        Qtot[k][0]=np.mean(temp,axis=0)
        if(len(temp)<=1):
            Qtot[k][1]=0
        else:
            Qtot[k][1]=np.std(temp,axis=0)/np.sqrt(len(temp)-1)
            

    #Qtot=-np.array(Qtot)

    fig=plt.figure(figsize=(4,10.0*combe12/6.0))
    plt.tick_params(labelbottom='off',labelleft='off')
    plt.axis('off')
    labSize=7
   

    ColMain='orange'
    if(evstype=='hel'):
        col1='black'
        col2='blue'
        if(ThreeHels): ColMain='red'
    
        labc0=r'$f_H=-1$'
        labc1=r'$f_H=0$'
        labc2=r'$f_H=1$'
        corr=0
   
    if(evstype=='k'):
        #stuff for when varying the other parameters.
        col1='orange'
        col2='green'
        ColMain='blue'
        corr=-1
        labc0=r'$k_{\mathtt{mag}}=0.01/\mathtt{Mpc}$'
        labc1=r'$k_{\mathtt{mag}}=0.05/\mathtt{Mpc}$'
        labc2=r'$k_{\mathtt{mag}}=0.1/\mathtt{Mpc}$'

    if(evstype=='B'):
        col1='orange'
        col2='green'
        ColMain='blue'
        corr=-1.5
        labc0=r'$B_{\mathtt{rms}}=1\times 10^{-14}~$G'
        labc1=r'$B_{\mathtt{rms}}=5\times 10^{-15}~$G'
        labc2=r'$B_{\mathtt{rms}}=2\times10^{-15}~$G'


   

    #plt.title('Q Statistics')
    if(ThreeHels):
        red_line = mlines.Line2D([], [], marker='None',color=ColMain, label=labc0)
        black_line = mlines.Line2D([], [], marker='None',color=col1, label=labc1,ls='dashed')
        blue_line = mlines.Line2D([], [], marker='None',color=col2, label=labc2,ls='dotted')

        plt.legend(handles=[red_line,black_line,blue_line],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #plt.legend(handles=[blue_line,red_line,black_line],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,fontsize=labSize+corr,handlelength=3)
    if(N==1 or plotallQ):
        fig.text(0.001, 0.5,r'$Q$ [$10^{-6}$]', va='center',rotation='vertical',fontsize=labSize+4)
    else:
        fig.text(0.001, 0.5,r'$\bar{Q}$ [$10^{-6}$]', va='center', rotation='vertical',fontsize=labSize+4)
    
   
    for k in range(combe12):
        temp=fig.add_subplot(combe12,1,k+1)
        if(plotallQ==True):
            ColMain='orange'
            for i in range(len(Q[k])):
                temp.plot(region,s*Q[k][i],color='blue') 
        temp.plot(region,s*Qtot[k][0],marker='None',color=ColMain,linewidth=linew,linestyle="-")
        if(plotallQ==False):
            temp.fill_between(region, s*Qtot[k][0]-Qtot[k][1], s*Qtot[k][0]+Qtot[k][1],alpha=0.5,facecolor=ColMain)
        if(ThreeHels):
            temp.plot(region,s*Qextratot[0][k][0],marker='None',color=col1,linewidth=linew,linestyle="dashed")
            temp.fill_between(region, s*Qextratot[0][k][0]-Qextratot[0][k][1], s*Qextratot[0][k][0]+Qextratot[0][k][1],alpha=0.5,facecolor=col1)
            temp.plot(region,s*Qextratot[1][k][0],marker='None',color=col2,linewidth=linew+1,linestyle="dotted")
            temp.fill_between(region, s*Qextratot[1][k][0]-Qextratot[1][k][1], s*Qextratot[1][k][0]+Qextratot[1][k][1],alpha=0.5,facecolor=col2)

        #temp.errorbar(region,Qtot[k][0],Qtot[k][1],color='orange')
        if(labelEbins):
            temp.set_ylabel(plotlabels[k],size=labSize,rotation=270,labelpad=15)
            temp.yaxis.set_label_position("right")
        
        if(k<combe12-1):
            temp.tick_params(axis='x',which='both',labelbottom='off',labelsize=labSize)
            temp.tick_params(axis='y',which='both',labelsize=labSize)
        else:
            temp.tick_params(which='both',labelsize=labSize)
    

    plt.xlabel('R [deg]',size=labSize+2)   
    #plt.tight_layout()
    path='sim_data/SimPlots/'
    if(not os.path.exists(path)):
        os.makedirs(path)
    if(plotinCN):
        fig.savefig(directory+'Qplot_'+name,dpi=150)
    else:
        fig.savefig('sim_data/QSimPlots/Q_'+name,dpi=150)
    plt.clf()
    plt.close() 

    return

def QfromEvs(evsnum,name='_',Qstart='Qv1_',CQ=0):
    njets=1
    for num in evsnum:
        QJets(NumJets=1,cn=num,UseUIBins=True,Qstart=Qstart,CQ=CQ)
    QsPlot(EvsList=evsnum,njets=njets,plotallQ=False,Qstart=Qstart,name=name)
    return

def Qhist(evsnum,name='_',Qstart='Q_',nbins=100,maxq=0,screwupsign=1):
    s=screwupsign
    directory='sim_data/3devents/case'+str(evsnum[0])+'/'
    combe12=len([filename for filename in os.listdir(directory) if filename.startswith(Qstart+'1_bin')])
    Q=[[] for j in range(combe12)]

    Ebins=np.loadtxt('sim_data/3devents/case'+str(evsnum[0])+'/UsedEbins')
    e12comb=list(combinations(range(len(Ebins)-1),2))

    if(np.loadtxt('sim_data/3devents/case'+str(evsnum[0])+'/BinsSelection')==0):#This uses UI bins
        plotlabels=[str(Ebins[e12comb[iloop][0]])+'-'+str(Ebins[e12comb[iloop][0]+1])+' GeV \n'+str(Ebins[e12comb[iloop][1]])+'-'+str(Ebins[e12comb[iloop][1]+1])+' GeV'  for iloop in range(combe12)]
    else: #Likely use AutoBins --- Unless more options were added.
        plotlabels=['Bins '+str(int(Ebins[e12comb[iloop][0]]))+'&'+str(int(Ebins[e12comb[iloop][1]]))  for iloop in range(combe12)]

    for i in evsnum:
        directory='sim_data/3devents/case'+str(i)+'/'
        Qfiles= [[directory+filename for filename in os.listdir(directory) if np.logical_and(filename.startswith(Qstart),filename.endswith('bin_'+str(k+1)))] for k in range(combe12)] #Qfiles[i] contains the Q values for the Ebin i+1 for all the jets.
        for j in range(combe12):
            for qfn in Qfiles[j]:
                temp=np.loadtxt(qfn)
                if(not np.dot(temp,temp)==0):#averaging over the Q's that deviates from 0 only.
                    if(maxq!=0):
                        if(np.abs(temp[-1])<maxq):
                            Q[j].append(s*temp[-1])#Values of all the last Qs
                    else: Q[j].append(s*temp[-1])#Values of all the last Qs

    bins=nbins
    fig=plt.figure(figsize=(4,10.0*combe12/6.0))
    plt.axis('off')
    labSize=4
     #plt.title(r'$Q(R_{\mathtt{max}})$ Counts')
    plt.tick_params(labelbottom='off',labelleft='off')
    
    fig.text(0.001, 0.5,r'$Q(R_{\mathtt{max}})$ Counts', va='center', rotation='vertical',size=labSize+5)

    for k in range(combe12):# range(combe12):
        temp=fig.add_subplot(combe12,1,k+1)
        temp.hist(Q[k],bins, facecolor='blue', alpha=0.75)
        temp.tick_params(labelsize=labSize+5)
        temp.set_ylabel(plotlabels[k],size=labSize+3,rotation=270,labelpad=20)
        temp.yaxis.set_label_position("right")
    plt.xlabel(r'$Q(R_{\mathtt{max}})$ [$10^{-6}$]',size=labSize+5)  
    
    plt.tight_layout()
     
    fig.savefig('sim_data/QSimPlots/Qhisto_'+name,dpi=150)
    plt.close() 

    return

   
def showfig(n,fig,njets=1,replotQ=False,replot3D=False,phHel=False,showQs=True,show3dplots=True):
    if(not show3dplots):
        showQs=False
    if(not showQs): 
        replotQ=False
        phHel=False
    if(phHel):njets=1
    res=200
    from scipy.misc import imread
    
    nplots=1
    images=['sim_data/3devents/case'+str(n)+'/jetskyimg_'+str(i+1)+'.png' for i in range(njets)]
    imgs=[imread(images[i]) for i in range(njets)]

    if(show3dplots):
        nplots=2
        imgs3dpath='sim_data/3devents/case'+str(n)+'/3dimg_'+str(njets)+'.png'
        if((not os.path.exists(imgs3dpath)) or replot3D):
            for i in range(njets):
                createplot3DEvs(n,i)
        imgs3dpath=['sim_data/3devents/case'+str(n)+'/3dimg_'+str(i+1)+'.png' for i in range(njets)]
        imgs3d=[imread(imgs3dpath[i]) for i in range(njets)]


    
    if(showQs):
        Qregion=np.loadtxt('sim_data/3devents/case'+str(n)+'/region_1')
    
        Qimgpath='sim_data/3devents/case'+str(n)+'/Qplot_SS.png'
        if(os.path.exists(Qimgpath) and (not replotQ)):
            imgsQ=imread(Qimgpath)
        else:
            QsPlot(N=1,cns=n,name='SS',njets=njets,plotinCN=True)
            imgsQ=imread(Qimgpath)
        nplots=3

    if(phHel): 
        nplots=4
        imageshel=['sim_data/3devents/case'+str(n)+'/BcB.png' for i in range(njets)]
        imgsphHel=[imread(imageshel[i]) for i in range(njets)]

    gs = gridspec.GridSpec(njets, nplots)     
        
    ax1=[plt.subplot(gs[i,0],axisbg='white') for i in range(njets)]
    if(show3dplots): ax2=[plt.subplot(gs[i,1],axisbg='white') for i in range(njets)]
    if(showQs): ax3=[plt.subplot(gs[i,2],axisbg='white') for i in range(njets)]
    if(phHel): ax4=[plt.subplot(gs[i,3],axisbg='white') for i in range(njets)]
    for i in range(njets):
        
        ax1[i].imshow(imgs[i])
        ax1[i].axis('off')
        ax1[i].tick_params(which='both',labelbottom='off',labelleft='off')

        if(show3dplots):
            ax2[i].imshow(imgs3d[i])
            ax2[i].axis('off')
            ax2[i].tick_params(which='both',labelbottom='off',labelleft='off')

        if(showQs):
            ax3[i].imshow(imgsQ)
            ax3[i].axis('off')
            ax3[i].tick_params(which='both',labelbottom='off',labelleft='off')
    
        if(phHel):
            ax4[i].imshow(imgsphHel[i])
            ax4[i].axis('off')
            ax4[i].tick_params(which='both',labelbottom='off',labelleft='off')

    
    plt.tight_layout()
    return


IMGINT=0
cnsSS=0
njetsSS=1
figSS=plt.figure(facecolor='white')

def onclick(event):
    global IMGINT,figSS,cnsINT,njetsSS,phHelSS
    if(event.key=='shift'):
        figSS.clear()
        IMGINT -= 1
        n=cnsSS+IMGINT
    else:
        figSS.clear()
        IMGINT += 1
        n=cnsSS+IMGINT
    print(n)
    showfig(n,figSS,njets=njetsSS,phHel=phHelSS,showQs=showQsSS,show3dplots=show3dplotsSS)
    plt.draw()
    return

def SSsky(cns,njets=1,replotQ=False,phHel=False,showQs=True,show3dplots=True):
    global IMGINT,cnsSS,figSS,njetsSS,phHelSS,showQsSS,show3dplotsSS
    phHelSS=phHel
    IMGINT=0
    cnsSS=cns
    njetsSS=njets
    showQsSS=showQs
    show3dplotsSS=show3dplots
    plt.close()
    figSS=plt.figure(facecolor='white')
    showfig(cnsSS,figSS,njets=njetsSS,replotQ=replotQ,phHel=phHelSS,showQs=showQsSS,show3dplots=show3dplotsSS)
    figSS.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return
