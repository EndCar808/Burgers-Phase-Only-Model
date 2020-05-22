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

def load_statistics(to_load,r_index,sim):
    length=len(to_load)
    flatness=np.zeros(length,dtype='float64')
    skewness=np.zeros(length,dtype='float64')
    alpha_list=np.zeros(length,dtype='float64')
    for j in range(length):
        i=to_load[j]
        name='_'+str(i)
        alpha=np.loadtxt(sim+'/parameters/Parameters' +name+'.dat')
        alpha=alpha[2]
        IS=np.loadtxt(sim+'/statistics/increment_statistics'+name+'.dat')
        moment2=IS[1,r_index]
        #moment2=np.loadtxt(sim+'/statistics/control_statistics.dat')
        #moment2=moment2[j,r_index]
        moment3=IS[2,r_index]
        moment4=IS[3,r_index]
        alpha_list[j]=alpha
        flatness[j]=moment4/moment2**2
        skewness[j]=moment3/np.sqrt(moment2)**3
    return alpha_list,skewness,flatness

def plot_transition(ax,to_load,r_index,title,NZ_list,name):
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel(r'$\alpha$',fontsize=labelsize)
    ax.set_yscale('log')   
    
    for N,Z,color,label in NZ_list:
        sim='N'+str(N)+'Z'+str(Z)
        alpha_list,skewness,flatness=load_statistics(to_load,r_index,sim)
        #ax.plot(alpha_list,skewness,color,label=label)
        ax.plot(alpha_list,flatness,color,label=label)
    ax.legend(fontsize=labelsize,loc=0,fancybox=True, framealpha=0.5)
    ax.plot(alpha_list,np.zeros(len(to_load)),'k--')
    ax.plot(alpha_list,3.0*np.ones(len(to_load)),'k--')
    ax.set_title('Flatness at the smallest scales for '+title,fontsize=labelsize)

def main_transition(to_load,r_index,title,NZ_list,name):
   
    fig = plt.figure(figsize=(5.,3.),facecolor = 'white')
    fig.subplots_adjust(top=0.90,bottom=0.15,left=0.07,right=0.97)
    ax  = fig.add_subplot(111)
    plot_transition(ax,to_load,r_index,title,NZ_list,name)
    fig.savefig(name+'transition'+str(r_index)+'.pdf',dpi=150)
    plt.show()

def main_two_transition():
    from matplotlib import gridspec
    gs=gridspec.GridSpec(1,2)
    
    fig = plt.figure(figsize=(10.,3.),facecolor = 'white')
    fig.subplots_adjust(top=0.89,bottom=0.16,left=0.05,right=0.98,wspace=.14)
    
    plt.figtext(.005,.92,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.49,.92,'b)',fontsize=labelsize,fontweight='bold')
    
    ax0=fig.add_subplot(gs[0,0])
    ax1=fig.add_subplot(gs[0,1])
       
    to_do=range(1,33)
    plot_transition(ax0,to_do,0,r'$N/k_0=2^8$.',[[10,4,'b',r'$\log_2N=$'+str(10)],[13,32,'g',r'$\log_2N=$'+str(13)],[15,128,'c',r'$\log_2N=$'+str(15)]],'NZ8_')
    plot_transition(ax1,to_do,0,r'$N=2^{15}$.',[[15,4,'r--',r'$k_0=$'+str(4)],[15,32,'k--',r'$k_0=$'+str(32)],[15,128,'c',r'$k_0=$'+str(128)]],'N15_')       

    fig.savefig('two_transition.pdf',dpi=150)
    plt.show()
       
def statistics(i,sim):
    name='_'+str(i)
    alpha=np.loadtxt(sim+'parameters/Parameters' +name+'.dat')
    alpha=alpha[2]
    CS=np.loadtxt(sim+'statistics/control_statistics.dat')
    labels=[r'$\langle\delta_ru\rangle^1$',r'$\langle\delta_ru\rangle^2$',r'$\langle\delta_ru\rangle^3$',r'$\langle\delta_ru\rangle^4$',r'$\langle\delta_ru\rangle^5$',r'$\langle\delta_ru\rangle^6$',r'$\langle\delta_ru\rangle^7$',r'$\langle\delta_ru\rangle^8$']
    x=np.loadtxt(sim+'realspace/RealSpaceCoordinates'+name+'.dat')
    N=int(x.size/2)
    x=x[1:N+1]
    IS=np.loadtxt(sim+'statistics/increment_statistics'+name+'.dat')
    return x, np.abs(IS[1]),alpha

def init_unnorm(i):
    
    fig = plt.figure(figsize=(10.,4.),facecolor = 'white')
    fig.subplots_adjust(top=0.94,bottom=0.13,left=0.08,right=0.99)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    alpha=0.0

    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    ax.set_ylabel(r'$\langle\delta_ru^2\rangle/L$',fontsize=labelsize)    # |: N^2
    for N,Z,color in [[10,4,'b'],[15,4,'r--'],[13,32,'g'],[15,32,'k--'],[15,128,'c']]:#['N10Z4','N13Z32','N15Z4','N15Z32','N15Z128']:
        sim='N'+str(N)+'Z'+str(Z)
        x,IS,alpha=statistics(i,sim+'/')
        int_len=2.0*np.pi/(1.0+Z)
        x/=int_len
        ax.plot(x,np.abs(IS)/int_len,color,label=r'$\log_2N=$'+str(N)+r', $k_0=$'+str(Z))
    #ax.plot(x,x*2,'b--',label=r'$\langle\delta_ru^2\rangle\sim r$')
    fig.suptitle(r'$\alpha=$'+str(np.around(alpha,decimals=5)),fontsize=labelsize)
    ax.legend(fontsize=labelsize,loc=2,fancybox=True, framealpha=0.5)
    ax.set_xlim([0.0001,10.])
    ax.set_ylim([10**-13,10**-6])
    ax.set_yscale('log')   
    ax.set_xscale('log')   
    fig.savefig('structure_functions_compare_unnorm_'+str(i)+'.pdf',dpi=150)
    #plt.show()
    #plt.close(fig)

def init(i):
    
    fig = plt.figure(figsize=(10.,4.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.11,left=0.08,right=0.99)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    alpha=0.0

    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    ax.set_ylabel(r'$\langle\delta_ru^2\rangle N^2/L$',fontsize=labelsize)    # |: N^2
    for N,Z,color in [[10,4,'b'],[15,4,'r--'],[13,32,'g'],[15,32,'k--'],[15,128,'c']]:#['N10Z4','N13Z32','N15Z4','N15Z32','N15Z128']:
        sim='N'+str(N)+'Z'+str(Z)
        x,IS,alpha=statistics(i,sim+'/')
        int_len=2.0*np.pi/(1.0+Z)
        x/=int_len
        ax.plot(x,np.abs(IS)*2.0**(2*N)/int_len,color,label=r'$\log_2N=$'+str(N)+r', $k_0=$'+str(Z))#*(2**(2*N))
    ax.plot([0.0001,10.],[0.0003,30.],'b--',label=r'$\langle\delta_ru^2\rangle N^2/L\sim r/L$')
    fig.suptitle(r'$\alpha=$'+str(np.around(alpha,decimals=5)),fontsize=labelsize)
    ax.legend(fontsize=labelsize,loc=4,fancybox=True, framealpha=0.5)
    ax.set_xlim([0.0001,10.])
    ax.set_ylim([0.0001,1])
    ax.set_yscale('log')   
    ax.set_xscale('log')   
    fig.savefig('structure_functions_compare_'+str(i)+'.pdf',dpi=150)
    #plt.show()
    #plt.close(fig)
    
if __name__=='__main__':
#    init(1)
#    init(13)
    init(21)
    init_unnorm(21)
#    init(24)
#    init(29)

    #to_do=range(1,33)
    #main_transition(to_do,0,r'$N/k_0=2^8$.',[[10,4,'b',r'$\log_2N=$'+str(10)],[13,32,'g',r'$\log_2N=$'+str(13)],[15,128,'c',r'$\log_2N=$'+str(15)]],'NZ8_')
    #main_transition(to_do,0,r'$N=2^{15}$.',[[15,4,'r--',r'$k_0=$'+str(4)],[15,32,'k--',r'$k_0=$'+str(32)],[15,128,'c',r'$k_0=$'+str(128)]],'N15_')
    #main_two_transition()