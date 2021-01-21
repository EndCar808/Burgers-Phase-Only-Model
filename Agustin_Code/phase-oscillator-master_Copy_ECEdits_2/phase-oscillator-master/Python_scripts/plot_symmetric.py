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

from plot_miguel import theoretical_energy

def load_parameters(base_dir='',
                    ):
    par=np.loadtxt(base_dir+'statistics/symmetric_parameters.dat')
    N=int(par[0])
    k0=int(par[1])
    alpha=float(par[2])
    beta=float(par[2])
    return N,k0,alpha,beta
    
def load_increment_symmetric(base_dir=''
                    ):
    r=np.loadtxt(base_dir+'statistics/coordinates_symmetric.dat')
    dur2=np.loadtxt(base_dir+'statistics/control_symmetric.dat')
    return r,dur2
    
def plot_control_symmetric(ax,
                base_dir='',
                style='r',
                label=''
                ):
    r,dur2=load_increment_symmetric(base_dir=base_dir)
    N,k0,alpha,beta=load_parameters(base_dir=base_dir)
    u2=theoretical_energy(N,k0,alpha,beta)
    dur2/=u2
    r/=np.pi/(k0+1)
    ax.plot(-r[:N],dur2[:N],'k--',label='Negative increments')
    ax.plot(r[N+1:],dur2[N+1:],'r',label='Positive increments')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([r[N+1],r[-1]])
    ax.set_ylim([dur2[N],np.max(dur2)])
    
def main_control_symmetric():

    fig = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig.subplots_adjust(top=0.97,bottom=0.12,left=0.13,right=0.97)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    
    ax.set_ylabel(r'$\langle\delta u_r^2\rangle /\langle u^2\rangle$',fontsize=labelsize)
    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    
    plot_control_symmetric(ax
                )
    ax.legend(fontsize=labelsize,loc=4,ncol=1,fancybox=True, framealpha=0.5)

    fig.savefig('figures/plot_ssymmetric.pdf',dpi=150)
                
if __name__ == '__main__':
    main_control_symmetric()