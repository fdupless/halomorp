import numpy as np
import sys,os
if(not sys.version_info[0]<3):
    from importlib import reload
import globalvars as gv
    
def calcqV2(e1vec,e2vec,e3vec,iE,jE,AngCutoff):
        Rmin=gv.MFPg(gv.zE(gv.dS),gv.EBins[iE+1])/gv.MFPg(gv.zE(gv.dS),gv.EBins[jE])
        Rmax=gv.MFPg(gv.zE(gv.dS),gv.EBins[iE])/gv.MFPg(gv.zE(gv.dS),gv.EBins[jE+1])
        if(len(e1vec)*len(e2vec)*len(e3vec)==0):
            return [np.zeros(len(gv.region)),np.zeros(len(gv.region))]
        # matrix with all the scalar products of e3 and e1 or e2
        dote1e3=np.dot(e3vec,e1vec.transpose())
        dote2e3=np.dot(e3vec,e2vec.transpose())
        qval=np.zeros(len(gv.region))
        qerr=np.zeros(len(gv.region))
        for l in range(len(gv.region)): #start from larger gv.region is faster
            rsize = np.cos(gv.region[len(gv.region)-1-l]*gv.degrad)
            # mask pairs that are too far away
            
            e2inregion = dote2e3 > rsize
            eta1=np.zeros((e3vec.shape[0],3)) # N_e3 values of eta1
            eta2=np.zeros((e3vec.shape[0],3)) # N_e3 values of eta2
            eta1c2=np.zeros((e3vec.shape[0],3))
            for t in range(len(eta1)):
                e2select=e2vec[e2inregion[t]]
                if (e2select.shape[0]==0):
                    eta1[t]=np.zeros(3)
                    eta2[t]=np.zeros(3)
                else:
                    for k in range(len(e2select)):
                        davg=np.sqrt(2*(1-np.dot(e2select[k],e3vec[t])))
                        rsizemax=max(np.cos(Rmax*davg),np.cos(gv.degrad*gv.maxangle))
                        rsizemin=np.cos(Rmin*davg)
                        e1inregion = (dote1e3[t]-rsizemax)*(rsizemin-dote1e3[t])>0
                        e1select=e1vec[e1inregion]
                        
                        select=[]
                        for p in range(len(e1select)):
                            norme1de3=np.sqrt(2*(1+np.dot(e1select[p],e3vec[t])))
                            test=np.dot(e2select[k]-e3vec[t],e1select[p]-e3vec[t])-davg*np.sqrt(2*(1+np.dot(e2select[k],e3vec[t])))-norme1de3*davg*np.cos(AngCutoff)
                            if(test>0):
                                select.append(True)
                            else:
                                select.append(False)
                        if(e1select[select].shape[0]==0):
                            eta1[t]=np.zeros(3)
                        else:
                            eta1[t]=np.average(e1select[select],axis=0)
                            #eta1[t]=np.average(e1vec[e1inregion],axis=0)
                        eta1c2[t]=eta1c2[t]+np.cross(eta1[t],e2select[k])
            #calculate cross product
            
                if (e2select.shape[0]==0):
                    eta1c2[t]=np.zeros(3)
                else:
                    eta1c2[t]=eta1c2[t]/(1.0*len(e2select))
                    #eta1c2=np.cross(eta1, eta2) # this is an array of N_e3 vectors
                
                #dot with eta3 and sum, then divide by n3. store in qval array
            eta1c2d3=np.diag(np.dot(e3vec,eta1c2.transpose()))
            qval[len(gv.region)-1-l]=eta1c2d3.mean()
            qerr[len(gv.region)-1-l]=eta1c2d3.std()
            	
# Divide the std in qerr by sqrt(N_e3)
        qerr/=np.sqrt(e3vec.shape[0])
        # Multiply everything by 10^6
        qval*=1.e6
        qerr*=1.e6
        return [qval,qerr]

def calcqV3(e1vec,e2vec,e3vec):
        if(len(e1vec)*len(e2vec)*len(e3vec)==0):
            return [np.zeros(len(gv.region)),np.zeros(len(gv.region))]
        # matrix with all the scalar products of e3 and e1 or e2
        dote1e3=np.dot(e3vec,e1vec.transpose())
        dote2e3=np.dot(e3vec,e2vec.transpose())
        qval=np.zeros(len(gv.region))
        qerr=np.zeros(len(gv.region))
        for l in range(len(gv.region)): #start from larger gv.region is faster
            rsize = np.cos(gv.region[len(gv.region)-1-l]*gv.degrad)
            # mask pairs that are too far away
            e1inregion = dote1e3 > rsize
            e2inregion = dote2e3 > rsize
            
            eta1=np.zeros((e3vec.shape[0],3)) # N_e3 values of eta1
            eta2=np.zeros((e3vec.shape[0],3)) # N_e3 values of eta2
            Se2ce1=np.zeros((e3vec.shape[0],3)) # N_e3 values of sum e2 cross eta1
            for t in range(len(eta1)):
                if (e1vec[e1inregion[t]].shape[0]==0 or 
                        e2vec[e2inregion[t]].shape[0]==0):
                    eta1[t]=np.zeros(3)
                    eta2[t]=np.zeros(3)
                    Se2ce1[t]=np.zeros(3)
                else:
                    e2vecinR=e2vec[e2inregion[t]]
                    e1vecA=e1vec[e1inregion[t]]
                    Se2ce1[t]=np.zeros(3)
                    for e2 in e2vecinR:
                        dote1e2=np.dot(e2[0:2],e1vecA[:,0:2].transpose()) > 0
                        e1LOSe2=e1vecA[dote1e2]
                        if(len(e1LOSe2)>0):
                            Se2ce1[t]=Se2ce1[t]-np.cross(e2,np.average(e1LOSe2,axis=0))
                        else:
                            Se2ce1[t]=np.zeros(3)
                    Se2ce1[t]=Se2ce1[t]/len(e2vecinR)
            
                #dot with eta3 and sum, then divide by n3. store in qval array
            eta1c2d3=np.diag(np.dot(e3vec,Se2ce1.transpose()))

            qval[len(gv.region)-1-l]=eta1c2d3.mean()
            qerr[len(gv.region)-1-l]=eta1c2d3.std()
            	
# Divide the std in qerr by sqrt(N_e3)
        qerr/=np.sqrt(e3vec.shape[0])
        # Multiply everything by 10^6
        qval*=1.e6
        qerr*=1.e6
        return [qval,qerr]
