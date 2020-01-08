#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:33:46 2018

@author: monvilre
"""
import matplotlib.ticker as ticker
from scipy import zeros
import numpy as np
import matplotlib as mp
mp.use('ps')
import os,sys
import matplotlib.pyplot as plt
import para


mp.rcParams.update({'font.size': 16}) 
plt.rcParams.update({'figure.autolayout': True})

#plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] =True


from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()




E = para.E # Nombre d'Ekman
P = para.P # Nombre de Prandt
S =  para.S # Nombre de Schmidt
L = S/P # Nombre de Lewis
#PrefBchim = 9.104556987060905
#PrefBtherm = 29.819405763030996
eta = para.eta
nbr =para.nbr# Nombre de points nbr*2.5
llog =para.llog # Limite en échelle log = 10^llog
llin = para.llin # Limite en échelle lin

M= para.M # Liste des modes 
LL = para.LL

b1 = np.logspace(llog,4,num= 3) # Ticks
b2 = np.logspace(4,llog,num= 3)
b3 = [0]
b = np.concatenate((-b1,b3,b2))


log = para.log # 0:Echelle linéaire 1:Echelle logarithmique
#remplir = para.remplir # 0:vide 1:Mode 2:Taux d'acroissement 3:Fréquence du mode



#def racine (m,Rc,Rt):
#            a = (np.sqrt(((l**2)*((np.pi)**2))+ m**2))
#            j=complex(0,1)
#            A = -j*(a**2)*(P**2)
#            B = -P**2*a**4-P*a**4/L-a**4*P-2*j*P**2*m/E
#            C = j*a**6/L-j*m**2*Rt*P+j*P*a**6/L-2*P*a**2/L*m/E-2*a**2*P*m/E+j*a**6*P-j*m**2*Rc*P
#            D = a**8/L+2*j*a**4/L*m/E-m**2*Rt*a**2/L-m**2*Rc*a**2
#            coeff = [A,B,C,D]
#            return np.roots(coeff)

def racine (m,Rc,Rt,l):
            a = (np.sqrt(((l**2)*((np.pi)**2))+ m**2))
            j=complex(0,1)
            A = -j*(a**2)*(P**2)
            B = -P**2*a**4-P*a**4/L-a**4*P-eta*j*P**2*m
            C = j*a**6/L-j*m**2*Rt*P+j*P*a**6/L-eta*P*a**2/L*m-eta*a**2*P*m+j*a**6*P-j*m**2*Rc*P
            D = a**8/L+eta*j*a**4/L*m-m**2*Rt*a**2/L-m**2*Rc*a**2
            coeff = [A,B,C,D]
            return np.roots(coeff)

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

class ProgressBar:
    iteration = 0
    total = 0
    prefix = ''
    suffix = ''
    decimals = 2
    barLength = 100
     
    usePercentage = True
     
    label = ''
     
    fillingChar = '='
    emptyChar = ' ' #-
    beginChar = '['
    endChar = ']'
     
    def __init__(self, iteration, total = 100, fillingChar = '=', emptyChar = ' ', beginChar = '[', endChar = ']', prefix = '', suffix = '', decimals = 2, barLength = 30, **kwargs):
        self.iteration = iteration
        self.total = total
        self.fillingChar = fillingChar
        self.emptyChar = emptyChar
        self.beginChar = beginChar
        self.endChar = endChar
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.barLength = barLength
        if kwargs.get("label") != None:
            self.label = kwargs.get("label")
 
        if kwargs.get("usePercentage") == False:
            self.usePercentage = False
        else:
            self.usePercentage = True
             
        self.updateProgress(iteration, self.label)
     
    def updateProgress(self, iteration, label):
 
        self.iteration = iteration
        self.label = label
        filledLength    = int(round(self.barLength * self.iteration / float(self.total)))
        percents        = round(100.00 * (iteration / float(self.total)), self.decimals)
        bar             = self.fillingChar * filledLength + self.emptyChar * (self.barLength - filledLength)
 
        sys.stdout.write("\r                                                                            ")
        if self.usePercentage:
            sys.stdout.write('\r%s %s%s%s %s%s %s' % (self.prefix, self.beginChar, bar, self.endChar, percents, '%', self.suffix)),
        else:
            sys.stdout.write('\r%s %s%s%s %s %s' % (self.prefix, self.beginChar, bar, self.endChar, label, self.suffix)),
        sys.stdout.flush()
        if self.iteration == self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()
##-------------------------------------
#'Grid'
##-------------------------
#l1 = np.logspace(llog,1,nbr)
#l2 = np.logspace(1,llog,nbr)
#l3=np.array([-10,-8,-6,-4,-2,2,4,6,8,10]) 
#RT = np.concatenate((-l1,l3,l2))
#Rat_all = []
#Rac_all = []
#for k in range(0,len(RT)):
#    T = RT[k]
#    l2log = (-10*T-(9.95*np.abs(T)))
##   
#    if np.abs(l2log) > 1:
#        if l2log > 0:
#            nombre = int(((llog - np.log10(l2log))/llog)*nbr)
#            rcplus = (np.logspace(np.log10(l2log),llog,nombre))
#        else:
#            nombre = int((np.log10(-l2log)/llog*nbr))
#            a = -np.logspace(np.log10(-l2log),1,nombre)
#            b = np.logspace(1,llog,nbr)
#            rcplus = np.concatenate((a,b))
#            nombre= nombre+nbr
#        Rac_all = np.concatenate((Rac_all,rcplus))
#        
#        for p in range(0,nombre):
#            K =[T]
#            Rat_all= np.concatenate((Rat_all,K))
#plt.scatter(Rac_all,Rat_all)
#plt.yscale('symlog')
#plt.xscale('symlog')
#plt.xlim(-1e29,1e29)
#plt.ylim(-1e29,1e29)
#plt.grid()
#plt.show()

trh = para.trh
if log == 1:
    l1 = np.logspace(llog,trh,nbr)
    l2 = np.logspace(trh,llog,nbr)
    l3= np.linspace(10000,10000,int(nbr/10)) 
    RC = np.concatenate((-l1,l3,l2))
    RT = np.concatenate((-l1,l3,l2))
elif log == 0:
    RC = np.linspace(-llin,llin,nbr)
    RT = np.linspace(-llin,llin,nbr+1)



mall = np.ones((1,np.size(LL))) * np.array(M).reshape((-1,1))
lall = np.ones((np.size(M),1)) * np.array(LL).reshape((1,-1))
mall = mall.flat[0:len(M)*len(LL)]
lall = lall.flat[0:len(M)*len(LL)]

Rat_all = np.ones((1,np.size(RC))) * np.array(RT).reshape((-1,1))
Rac_all = np.ones((np.size(RT),1)) * np.array(RC).reshape((1,-1))



par_idx0 = int((mpi_rank/mpi_size)*np.size(Rac_all))
par_idx1 = int(((mpi_rank+1)/mpi_size)*np.size(Rac_all))

comm.Barrier

print("o Process %d Optimizing %d parameter sets (out of %d)" % (mpi_rank, (par_idx1-par_idx0)*len(mall), np.size(Rat_all)*len(mall)))

comm.Barrier()

Rac2_vec = Rac_all.flat[par_idx0:par_idx1]
Rat2_vec = Rat_all.flat[par_idx0:par_idx1]





#----------------------------------
'Minimum onset' 
#---------------------------------

sig= zeros(len(mall))
omeg= zeros(len(mall))


RCC,RTT=np.meshgrid(RC,RT)
MODE,LAM=np.meshgrid(RC,RT)
OMEG,MODL = np.meshgrid(RC,RT)

if mpi_rank == 0:
    print("""\n------------------------------------------------------
\t Mapping
------------------------------------------------------
""")
    print("Loading...")

result_all = np.zeros((0,6))

h = 1
Bar = None

for Rc,Rt in zip(Rac2_vec, Rat2_vec):
    c=0
    if mpi_rank == 0:
        label = "Loading"

        if Bar == None:
            Bar = ProgressBar(h, len(Rac2_vec), label = label, usePercentage = True)
        else:
            Bar =  ProgressBar.updateProgress(self =Bar,iteration =h, label= label)
        h = h+1
    
    for m,l in zip(mall, lall):         
        rac=racine(m,Rc,Rt,l)
        indice=np.argmax((-rac).imag)
        sig[c]= -rac[indice].imag
        omeg[c]= np.abs(rac[indice].real)
        c=c+1
   
    
    lamb =np.max(sig) 
    
    if lamb < 0 :
        mazim = np.nan
        mrad =np.nan
        omega = np.nan
    else:
        mazim = mall[np.argmax(sig)]
        mrad = lall[np.argmax(sig)]
        omega = omeg[np.argmax(sig)]
    
    result = np.transpose(np.vstack((Rc,Rt,mazim,lamb,omega,mrad)))
    result_all = np.vstack((result_all,result))
np.savetxt('Res_%d.txt' % mpi_rank, result_all)

comm.Barrier()

if mpi_rank == 0:
    result_all = np.zeros((0,6))
    for indmpi in range(0, mpi_size):
        resulti = np.loadtxt('Res_%d.txt' % indmpi)
        result_all = np.vstack((result_all, resulti))
    np.savetxt('Res.txt', result_all)    
    
    result_all = np.loadtxt('Res.txt')
    print('Done')


    RCC,RTT =np.meshgrid(RC,RT)
    Z =np.reshape(result_all[:,3],(len(RC),len(RC)))
    MODE =np.reshape(result_all[:,2],(len(RC),len(RC)))
    OMEG = np.reshape(result_all[:,4],(len(RC),len(RC)))
    OMEG = np.abs(OMEG)
    MODL = np.reshape(result_all[:,5],(len(RC),len(RC)))
    LAM = np.reshape(result_all[:,3],(len(RC),len(RC)))
    os.system('rm Res*.txt')   
    
    plt.figure(5)
    CS = plt.contour(RCC,RTT,Z,[0],colors='m')
    ligne = CS.allsegs[0][0]
    
    plt.figure(1)
    RC= -np.logspace(llog,4.5,8)
    plt.plot(RC,-RC,'--k')
    RC= np.logspace(llog,4.5,8)
    plt.plot(RC,(-RC),'--k')
    
    
    CM = plt.contour(RCC,RTT,MODE,M,cmap=plt.cm.viridis,norm = mp.colors.LogNorm())
    plt.colorbar(CM,label ='$m$',format =  r'$%d$')
    
    plt.figure(2)
    RC= -np.logspace(llog,4.5,8)
    plt.plot(RC,-RC,'--k')
    RC= np.logspace(llog,4.5,8)
    plt.plot(RC,(-RC),'--k')
    CM = plt.contourf(RCC,RTT,MODL,LL,cmap=plt.cm.winter,norm = mp.colors.LogNorm())
    plt.colorbar(CM,label ='$l$',format =  r'$%d$')
    
    
    plt.figure(3)
    RC= -np.logspace(llog,4.5,8)
    plt.plot(RC,-RC,'--k')
    RC= np.logspace(llog,4.5,8)
    plt.plot(RC,(-RC),'--k')
    bar = np.logspace(np.log10(np.nanmin(OMEG)),np.log10(np.nanmax(OMEG)),60)
    CO = plt.contourf(RCC,RTT,OMEG,bar,cmap=plt.cm.plasma,norm = mp.colors.LogNorm())
    plt.colorbar(CO,label ='$\omega$',ticks=[1,10,1e2,1e3,1e4,1e5,1e6])
    
    plt.figure(4)
    RC= -np.logspace(llog,4.5,8)
    plt.plot(RC,-RC,'--k')
    RC= np.logspace(llog,4.5,8)
    plt.plot(RC,(-RC),'--k')
    bar = np.logspace(-1,np.log10(np.max(LAM)),60)
    CL = plt.contourf(RCC,RTT,LAM,bar,cmap=plt.cm.spring,norm = mp.colors.LogNorm())
    plt.colorbar(CL,label ='$\sigma$',ticks=[1e-1,1,10,1e2,1e3,1e4,1e5,1e6])
    
    

comm.Barrier()
if mpi_rank == 0 :

    #---------------------
    'Modes at the onset'
    #--------------------
    

    lignemod =np.zeros((len(ligne)-1,2))
    
    p = np.zeros(len(ligne)-1)
    t = np.zeros(len(ligne)-1)
    sig= np.zeros(len(mall))
    

    print("""\n------------------------------------------------------
\t Instable Modes at the Onset
------------------------------------------------------
    """)
    print('Loading...')
    h = 1
    Bar = None
    for x in range(0,len(ligne)-1):
        label = "Loading"

        if Bar == None:
            Bar = ProgressBar(h, len(ligne)-1, label = label, usePercentage = True)
        else:
            Bar =  ProgressBar.updateProgress(self =Bar,iteration =h, label= label)
        h = h+1
        
        Rc=ligne[x+1,0] 
        Rt=ligne[x+1,1]
        c = 0
        for m,l in zip(mall, lall):         
                rac=racine(m,Rc,Rt,l)
                indice=np.argmax((-rac).imag)
                sig[c]= -rac[indice].imag
                c = c+1
                D = 0.001
        if np.max(sig) > 0:
            lignemod[x,0] = Rc
            lignemod[x,1] = Rt
        else:
            while np.max(sig) < 0:
            
                xA = ligne[x,0]
                xB = ligne[x+1,0]
                yA = ligne[x,1]
                yB = ligne[x+1,1]
                delta = D*np.sqrt(xB**2+yB**2)
                if np.abs((yB-yA)/yB) < 1e-2:
                    yC = yB + delta
                    xC = xB
                elif np.abs((xB-xA)/xB) < 1e-3:
                    yC = yB 
                    xC = xB +delta
                else:
                    alpha = -((xB-xA)/(yB-yA))
                    beta = (xB*(xB-xA)+yB*(yB-yA))/(yB-yA)
                    A = 1+alpha**2
                    B = -(2*xB)+2*alpha*(beta-yB)
                    C = -((delta)**2)+(xB**2)+((beta-yB)**2)
                    coef = [A,B,C]
                    solu = np.roots(coef)
                    xC = solu[0]
                    yC = alpha*xC+beta
                    vect = (xB-xA)*(yC-yB)-(yB-yA)*(xC-xB)
                    if vect < 0:
                        xC = solu[1]
                        yC = alpha*xC+beta
            
                lignemod[x,0] = xC
                lignemod[x,1] = yC
                Rc2=lignemod[x,0] 
                Rt2=lignemod[x,1]
                c = 0
                for m,l in zip(mall, lall):         
                    rac=racine(m,Rc2,Rt2,l)
                    indice=np.argmax((-rac).imag)
                    sig[c]= -rac[indice].imag
                    c = c+1
                D = D*1.5
        
        mazim = mall[np.argmax(sig)]
        mrad = lall[np.argmax(sig)]
        p[x] = mazim
        t[x] = mrad
    #sc = plt.scatter( lignemod[:,0],lignemod[:,1], c=np.log10(p[:]), vmin=np.log10(np.min(p)), vmax=np.log10(np.max(p)), s=15, cmap='cool')
    
    #plt.colorbar(sc)


    mini = np.zeros((len(p),4))
    mini[:,0] = lignemod[:,0]
    mini[:,1] = lignemod[:,1]
    mini[:,2] = p[:]
    mini[:,3] = t[:]
    if np.max(mini[:,3]) == np.max(lall):
        print('Warning !! saturation on radial wave number')
    if np.max(mini[:,2]) == np.max(mall):
        print('Warning !! saturation on azimuthal wave number')
    
    G = para.Gamma
    output = './E={},P={},S={},G={}'.format(E,P,S,G)
    if not os.path.exists(output):
        os.makedirs(output)
    
   
    np.savetxt(output + '/minimML.txt',mini)
    np.savetxt(output + '/RCC.txt',RCC)
    np.savetxt(output + '/RTT.txt',RTT)
    np.savetxt(output + '/MODE.txt',MODE)
    np.savetxt(output + '/MODL.txt',MODL)
    np.savetxt(output + '/OMEG.txt',OMEG)
    np.savetxt(output + '/LAM.txt',LAM)
    
    
    
    
    plt.figure(1)
    plt.yscale('symlog', linthreshy=10000)
    plt.xscale('symlog', linthreshy=10000)
    plt.xticks(b)
    plt.yticks(b)
    plt.xlabel('$Ra_C$')
    plt.ylabel('$Ra_T$')  
    
    plt.savefig(output + '/MODE.eps', format = 'eps') 
    
    plt.figure(2)
    plt.yscale('symlog', linthreshy=10000)
    plt.xscale('symlog', linthreshy=10000)
    plt.xticks(b)
    plt.yticks(b)
    plt.xlabel('$Ra_C$')
    plt.ylabel('$Ra_T$')
    plt.savefig(output + '/MODL.eps', format = 'eps') 
    
    plt.figure(3)
    plt.yscale('symlog', linthreshy=10000)
    plt.xscale('symlog', linthreshy=10000)
    plt.xticks(b)
    plt.yticks(b)
    plt.xlabel('$Ra_C$')
    plt.ylabel('$Ra_T$')
    plt.savefig(output + '/OMEG.eps', format = 'eps') 
    
    plt.figure(4)
    plt.yscale('symlog', linthreshy=10000)
    plt.xscale('symlog', linthreshy=10000)
    plt.xticks(b)
    plt.yticks(b)
    plt.xlabel('$Ra_C$')
    plt.ylabel('$Ra_T$')
    plt.savefig(output + '/LAMB.eps', format = 'eps')
   
    
    
    print('Done')

  
    #plt.savefig('.eps')

