#import matplotlib
#matplotlib.use('Agg')  # no UI backend
from scipy.io import FortranFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
labelsize=18
ticksize=15
ki='int64'
kr='float64'
kc='complex128'
ratio=2**8

def load_triads(total_triads,name):
    f_exp1=FortranFile('./statistics/trig_moment{}.dat'.format(name), 'r')
    exp1=f_exp1.read_reals(kc).reshape((total_triads), order="F")
    f_exp1.close
    return exp1
    
def plot_triangle(ax,k0,name,log=False):
    total_triads=k0**2*(ratio**2//4-ratio+1)
    N=ratio*k0
    exp1=load_triads(total_triads,name)
    k_list,p_list=TriadList(k0,ratio)

    if log==False:
        exp1_matrix=np.zeros([N//2+1,N+1],dtype='float64')
        exp1_matrix[p_list,k_list]=np.abs(exp1)
        exp1_matrix=np.where(exp1_matrix==0.0,np.nan,exp1_matrix)
        exp1_matrix=np.flipud(exp1_matrix)
        del k_list,p_list
        
        ax.set_xticks([0,N//2,N])
        ax.set_xticklabels([r'$0$',r'$N/2$',r'$N$'],fontsize=labelsize)
        ax.set_yticks([0,N//2])
        ax.set_yticklabels([r'$0$',r'$N/2$'],fontsize=labelsize)
        ax.set_ylabel(r'$p$', fontsize=labelsize)
        ax.set_xlabel(r'$k$', fontsize=labelsize)
        im=ax.imshow(exp1_matrix,interpolation='none',extent=[0,N,0,N/2],vmin=0.0,vmax=0.75,cmap='rainbow')
    else:
        import numpy.ma as ma
        exp1_matrix=np.zeros([N//2-k0,N-k0],dtype='float64')
        exp1_matrix[p_list-1-k0,k_list-1-k0]=np.abs(exp1)
        exp1_matrix=np.where(exp1_matrix==0.0,np.nan,exp1_matrix)
        exp1_matrix=ma.masked_array(exp1_matrix,mask=np.isnan(exp1_matrix))
        
        k_list=np.linspace(k0+1,N,num=N-k0)
        p_list=np.linspace(k0+1,N//2,num=N//2-k0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$p$', fontsize=labelsize)
        ax.set_xlabel(r'$k$', fontsize=labelsize)
        ax.set_xlim([2*(k0+1),k_list[-1]])
        ax.set_ylim([k0+1,p_list[-1]])
        im=ax.pcolormesh(k_list,p_list,exp1_matrix,vmin=0.0,vmax=.75,cmap='rainbow')
        del k_list,p_list
    return im,exp1
    
def plot_single_triad(k0,name):
    fig = plt.figure(figsize=(6.,3.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.24,left=0.16,right=1.00)
    ax=fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)

    
    fig1 = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig1.subplots_adjust(top=0.99,bottom=0.15,left=0.16,right=0.99)
    ax1=fig1.add_subplot(111)

    fig2 = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig2.subplots_adjust(top=0.99,bottom=0.15,left=0.16,right=0.99)
    ax2=fig2.add_subplot(111)

    im,exp1=plot_triangle(ax,k0,name)
    cb=fig.colorbar(im,orientation="vertical",ax=ax)
    cb.ax.tick_params(labelsize=ticksize)    
    cb.set_label(r'$R_{k,p}$', fontsize=ticksize)
    cb.set_ticks([0.0,0.5,1.0])
    cb.set_ticklabels([r'$0$',r'$\frac{1}{2}$',r'$1$'])#,fontsize=labelsize)

    ax1.hist(np.abs(exp1),np.linspace(0.0,1.0,num=50))
    ax2.hist(np.angle(exp1),np.linspace(-np.pi,np.pi,num=50))
    fig.savefig('figures/Triad{}.pdf'.format(name),dpi=320)

    return np.average(exp1)
    
def plot_two_triad():
    from matplotlib import gridspec
    gs=gridspec.GridSpec(1,2)
    log=True
    
    fig = plt.figure(figsize=(10.,2.5),facecolor = 'white')
    fig.subplots_adjust(top=0.88,bottom=0.23,left=0.08,right=1.00,wspace=0.24)
    ax0=fig.add_subplot(gs[0,0]) 
    ax0.tick_params(labelsize=ticksize)
    ax1=fig.add_subplot(gs[0,1]) 
    ax1.tick_params(labelsize=ticksize)
    plt.figtext(.01,.9,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.51,.9,'b)',fontsize=labelsize,fontweight='bold')
    ax0.set_title('Gaussian cutoff',fontsize=labelsize)
    ax1.set_title('Pure power law',fontsize=labelsize)

    im0,exp0=plot_triangle(ax0,4,1,log=log)
    im1,exp1=plot_triangle(ax1,4,2,log=log)

    cb0=fig.colorbar(im0,orientation="vertical",ax=ax0)
    cb0.ax.tick_params(labelsize=ticksize)    
    cb0.set_label(r'$R_{k,p}$', fontsize=ticksize)
    cb0.set_ticks([0.0,0.5,0.75])
    cb0.set_ticklabels([r'$0$',r'$\frac{1}{2}$',r'$\frac{3}{4}$'])

    cb1=fig.colorbar(im1,orientation="vertical",ax=ax1)
    cb1.ax.tick_params(labelsize=ticksize)    
    cb1.set_label(r'$R_{k,p}$', fontsize=ticksize)
    cb1.set_ticks([0.0,0.5,0.75])
    cb1.set_ticklabels([r'$0$',r'$\frac{1}{2}$',r'$\frac{3}{4}$'])#,fontsize=labelsize)
    if log==True:
        fig.savefig('figures/Two_Triad.jpeg',dpi=320)
    else:
        fig.savefig('figures/Two_Triad.pdf',dpi=320)
    
def main():
    total_runs=2
    triad_average=np.zeros(8,dtype='float64')
    alpha = np.zeros(8, dtype='float64')
    beta = np.zeros(8, dtype='float64')
    for name in range(1,total_runs+1):
        triad_average[name-1]=np.abs(plot_single_triad(4,name))
        alpha_l = np.loadtxt('parameters/Parameters_' + str(name) + '.dat').flatten()
        beta[name-1] = alpha_l[3]
        alpha[name-1] = alpha_l[2]

    fig = plt.figure(figsize=(6.,3.5),facecolor = 'white')
    fig.subplots_adjust(top=1.00,bottom=0.21,left=0.18,right=0.97)
    ax=fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    #ax.set_xticks([0,N//2,N])
    #ax.set_xticklabels([r'$0$',r'$N/2$',r'$N$'],fontsize=labelsize)
    #ax.set_yticks([0,N//2])
    #ax.set_yticklabels([r'$0$',r'$N/2$'],fontsize=labelsize)
    ax.set_ylabel(r'$\langle\vert R_{k,p}\vert\rangle$', fontsize=labelsize)
    ax.set_xlabel(r'$\alpha$', fontsize=labelsize)
    ax.plot(alpha,triad_average)
    fig.savefig('figures/Triad_average.jpeg', dpi=320)

    plt.show()
        
def TriadList(k0,ratio):
    N=ratio*k0
    total_triads=k0**2*(ratio**2//4-ratio+1)
    l=0
    k_list=np.empty(total_triads,dtype=np.int)
    p_list=np.empty(total_triads,dtype=np.int)
    for p in range(k0+1,N//2+1):
        for k in range(2*p,N+1):
            k_list[l]=k
            p_list[l]=p
            l=l+1
    return k_list,p_list

if __name__ == '__main__':
    plot_two_triad()
