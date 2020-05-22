#Agustin Arguedas

import numpy as np
#import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

labelsize=15
ticksize=12

from plot_miguel import load_parameters

def load_parameters(base_dir='',
                    name='1'
                    ):
    par=np.loadtxt(base_dir+'parameters/Parameters_'+name+'.dat')
    N=int(par[0])
    k0=int(par[1])
    alpha=float(par[2])
    beta=float(par[3])
    return N,k0,alpha,beta
    
def a_k(N,
        k0,
        alpha,
        beta,
        k
        ):
    k_d=np.arange(k0+1,N+1,dtype='float64')
    return k_d,k_d**(-alpha)*np.exp(-beta*(2.0*k_d/float(N))**2),k,k**(-alpha)*np.exp(-beta*(2.0*k/float(N))**2)
    
def plot_spectrum(ax,
                base_dir='',
                name='1',
                style='r',
                label=''
                ):
    N,k0,alpha,beta=load_parameters(base_dir=base_dir,name=name)
    k_d,ak_d,k,ak=a_k(N,
            k0,
            alpha,
            beta,
            k=np.linspace(k0+1,N,num=128*N,dtype='float64'))
    #ax.plot(k,ak**2,'k--',label=label)
    ax.plot(k_d,ak_d**2,style,label=label)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([k0*0.5,N*2])
    
def main_spectrum():

    fig = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.12,left=0.13,right=0.97)
    N,k0,alpha,beta=load_parameters()
    fig.suptitle(r'$N={0}, k_0={1}, \alpha={2}, \beta={3}$'.format(N,k0,alpha,beta),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    
    ax.set_ylabel(r'$E(k)$',fontsize=labelsize)
    ax.set_xlabel(r'$k$',fontsize=labelsize)
    
    plot_spectrum(ax
                )
    ax.legend(fontsize=labelsize,loc=3,ncol=2,fancybox=True, framealpha=0.5)

    fig.savefig('figures/plot_spectrum.pdf',dpi=150)
                
if __name__ == '__main__':
    main_spectrum()