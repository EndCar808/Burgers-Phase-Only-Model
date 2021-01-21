#Agustin Arguedas

from scipy.io import FortranFile
kr='float64'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

labelsize=15
ticksize=12

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utilities import load_parameters, u_rms, load_RS, load_stats

def plot_miguel(ax,
                name,
                moments_to_plot=[2,4],
                labels=[r'$S_2(r)$',r'$S_4(r)$'],
                style=['r','g'],
                L_int=True,
                control=False,
                base_dir=''
                ):
    pars=load_parameters(name=name,base_dir=base_dir)
    x=load_RS(name,'RealSpaceCoordinates')
    x=x[1:pars['N']+1]
    if L_int==True:
        int_len=np.pi/(pars['k0']+1.0)
        x/=int_len
    urms=u_rms(N=pars['N'],k0=pars['k0'],alpha=pars['alpha'],beta=pars['beta'])
    subname='increment_statistics'
    IS=load_stats(name,subname,base_dir=base_dir)
    IS=IS.reshape(pars['N'],pars['n_max']).T
    for moment,label,line in zip(moments_to_plot,labels,style):
        ax.plot(x,np.abs(IS[moment-1])/urms**moment,line,label=label)
    if control == True:
        ax.plot(x,np.loadtxt(base_dir+'statistics/control_simple.dat'),'k--')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([x[0],1.0])
    ax.set_ylim([2*10**-6,35])

def plot_zeta_p(ax,
                name,
                moments_to_plot=[2,4],
                labels=[r'$S_2(r)$',r'$S_4(r)$'],
                style=['r','g'],
                L_int=True,
                base_dir=''
                ):
    from utilities import get_log_derivative
    x=load_RS(name,'RealSpaceCoordinates')
    for moment,label,line in zip(moments_to_plot,labels,style):
        px,py=get_log_derivative(name,
                                 moment=moment,
                                 normed=False,
                                 base_dir=base_dir
                       )
        ax.plot(px,py,line,label=label)
        x0_py=(py[1]-py[0])/(np.log10(px[1])-np.log10(px[0]))*(np.log10(x[0])-np.log10(px[0]))+py[0]
        x0_py=(py[1]-py[0])/(px[1]-px[0])*(x[0]-px[0])+py[0]
        print(x0_py)
        ax.plot([x[0],px[0]],[x0_py,py[0]],'k--')
    ax.plot([px[0],1.0],[1.0,1.0],'k--')
    ax.set_xscale('log')
    #ax.set_xlim([x[0],x[-1]*2.0])
    ax.set_xlim([x[0],1.0])
    ax.set_ylim([0,2])#moments_to_plot[-1]])
    ax.tick_params(labelsize=ticksize-5)
    
def main_miguel(name):
    moments_to_plot=[2,3,4,5,6]
    labels=[r'$p=2$',r'$p=3$',r'$p=4$',r'$p=5$',r'$p=6$']
    style=['r','g','b','y','c']
    base_dir='../'
#    moments_to_plot=[2,4,6]
#    labels=[r'$p=2$',r'$p=4$',r'$p=6$']
#    style=['r','g','b']

    fig = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.12,left=0.13,right=0.97)
    pars=load_parameters(name=name,base_dir=base_dir)
    fig.suptitle(r'$N={0}, k_0={1}, \alpha={2}, \beta={3}$'.format(pars['N'],pars['k0'],pars['alpha'],pars['beta']),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    
    ax.set_ylabel(r'$\vert S_p(r)\vert/{u_\mathrm{rms}}^p$',fontsize=labelsize)
    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    
    plot_miguel(ax,
                name,
                moments_to_plot=moments_to_plot,
                labels=labels,
                style=style,
                base_dir=base_dir
                )
    ax.legend(fontsize=labelsize,loc=2,ncol=1,fancybox=True, framealpha=0.5)
            
    axins = inset_axes(ax, width=1.6, height=1.3,loc=4)#10,bbox_to_anchor=[390.,105])
    axins.set_ylabel(r'$\zeta_p$',fontsize=labelsize)

    plot_zeta_p(axins,
                name,
                moments_to_plot=moments_to_plot,
                labels=labels,
                style=style,
                base_dir=base_dir
                )
    
    fig.savefig('../figures/plot_miguel_{0}.png'.format(name),dpi=150)
    plt.show()
                
if __name__ == '__main__':
    for k0 in [0,5,7]:
        for i in range(1,17):
            main_miguel('N15_k{0}_{1}'.format(k0,i))
