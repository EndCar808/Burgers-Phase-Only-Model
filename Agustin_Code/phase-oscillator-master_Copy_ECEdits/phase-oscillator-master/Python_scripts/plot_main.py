#Agustin Arguedas

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from plot_routines import plot_flatness_alpha, plot_scaling_exponent, plot_Hists, plot_statistics, plot_flatness_derivative, plot_skewness, plot_flatness, labelsize,plot_flatness_slope, get_slope

def main_plot_statistics(i):
    name='_'+str(i)
    alpha=np.loadtxt('parameters/Parameters' +name+'.dat')
    alpha=alpha[2]
    fig = plt.figure(figsize=(12.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.12,left=0.10,right=0.79)
    fig.suptitle(r'$\alpha=$'+str(np.around(alpha,decimals=5)),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    plot_statistics(ax,i)
    ax.legend(fontsize=labelsize,loc=1,bbox_to_anchor=(1.32,1.))
    fig.savefig('figures/structure_functions'+name+'.pdf',dpi=150)
    plt.show()
    
def main_flatness_alpha(to_do):

    fig = plt.figure(figsize=(10.,5.),facecolor = 'white')
    fig.subplots_adjust(top=0.96,bottom=0.14,left=0.08,right=0.99)
    #fig.suptitle(r'$\alpha=$'+str(np.around(alpha,decimals=5)),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    plot_flatness_alpha(ax,to_do)
    ax.legend(fontsize=labelsize,loc=0,fancybox=True, framealpha=0.5)
    fig.savefig('figures/flatness_alpha.pdf',dpi=150)
    plt.show()
    #plt.close(fig)
    
def main_plot_flatness_derivative():
    fig = plt.figure(figsize=(10.,8.),facecolor = 'white')
    fig.subplots_adjust(top=0.905,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
    ax0=fig.add_subplot(111)
    to_do=[1,5,13,21,24,29]
    plot_flatness_derivative(ax0,to_do)

def main_Hists(i):
    name=str(i)

    fig = plt.figure(figsize=(12.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.91,bottom=0.12,left=0.10,right=0.79)
    ax  = fig.add_subplot(111)
    plot_Hists(ax,name,title=True)
    ax.legend(fontsize=labelsize,loc=1,bbox_to_anchor=(1.32,1.))      
    fig.savefig('figures/Hists'+name+'.pdf',dpi=150)
    #plt.show()
    plt.close(fig)
    
def main_hists_flatness():
    from matplotlib import gridspec
    gs=gridspec.GridSpec(3,3)
    
    fig = plt.figure(figsize=(10.,8.),facecolor = 'white')
    fig.subplots_adjust(top=0.905,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
    
    plt.figtext(.01,.92,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.92,'b)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.92,'c)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.01,.6,'d)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.6,'e)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.6,'f)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.05,.3,'g)',fontsize=labelsize,fontweight='bold')    
    
    ax0=fig.add_subplot(gs[0,0]) 
    ax1=fig.add_subplot(gs[0,1]) 
    ax2=fig.add_subplot(gs[0,2]) 
    ax3=fig.add_subplot(gs[1,0]) 
    ax4=fig.add_subplot(gs[1,1]) 
    ax5=fig.add_subplot(gs[1,2]) 
    ax6=fig.add_subplot(gs[2,:]) 
    
    to_do=[1,5,13,21,24,29]
    plot_flatness_alpha(ax6,to_do)
    ax6.legend(fontsize=labelsize,loc=1,ncol=2,fancybox=True, framealpha=0.5) 

    for ax,i in zip([ax0,ax1,ax2,ax3,ax4,ax5],to_do):
        plot_Hists(ax,str(i),title=True,Norm=True)
    ax1.legend(fontsize=labelsize,bbox_to_anchor=(-.25,1.2,1.5,0.25), loc=3,ncol=3, mode="expand", borderaxespad=0.,fancybox=True, framealpha=0.5)
    fig.savefig('figures/hists_flatness.pdf',dpi=150)
    plt.show()
    
def main_hists_flatness_scalexp(name,to_do):
    from matplotlib import gridspec
    from utilities import load_parameters, get_log_derivative
    gs=gridspec.GridSpec(4,3)
    
    fig = plt.figure(figsize=(12.,10.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
    
    plt.figtext(.01,.93,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.93,'b)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.93,'c)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.01,.68,'d)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.68,'e)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.68,'f)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.01,.45,'g)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.45,'h)',fontsize=labelsize,fontweight='bold')
    #plt.figtext(.01,.22,'i)',fontsize=labelsize,fontweight='bold')    
    
    ax0=fig.add_subplot(gs[0,0]) 
    ax1=fig.add_subplot(gs[0,1]) 
    ax2=fig.add_subplot(gs[0,2]) 
    ax3=fig.add_subplot(gs[1,0]) 
    ax4=fig.add_subplot(gs[1,1]) 
    ax5=fig.add_subplot(gs[1,2]) 
    ax6=fig.add_subplot(gs[2:,:2]) 
    ax7=fig.add_subplot(gs[2:,2]) 
    #ax8=fig.add_subplot(gs[3,:2])     

    plot_flatness_slope(name,ax7,list(range(1,17)))
    for i in [11,13]:
        pars=load_parameters(name=name+'_{0}'.format(i))
        alpha=pars['alpha']
        x,sl=get_log_derivative(name,
                       i=i,
                       moment=4,#Set this here by hand!
                       normed=False,
                       base_dir='../'
                       )#get_slope(l_name,i,moment=moment[j],normed=normed,scale=scale,base_dir=base_dir)
        slope=sl[16]
        #slope,intercept=get_slope(i,normed=True)
        ax7.plot([0.0,alpha,alpha],[slope,slope,0],'k--')
        x=x/(2.0*np.pi/pars['k0'])
    #ax7.set_ylim([-0.9,0.0])
    plot_flatness_alpha(name,ax6,to_do,scaling=False)
    print(x[16])
    ax6.plot([x[16],x[16]],[1.0,10000],'k--')
    ax6.legend(fontsize=labelsize,loc=1,ncol=2,fancybox=True, framealpha=0.5) 
    #plot_flatness_alpha(name,ax8,to_do,compensated=True)
    #ax8.legend(fontsize=labelsize,loc=2,ncol=2,fancybox=True, framealpha=0.5)

    
    for ax,i in zip([ax0,ax1,ax2,ax3,ax4,ax5],to_do):
        plot_Hists(ax,name+'_{0}'.format(i),title=True,Norm=True,moment=4)
        ax.set_ylim([10**-6,15.0])
    ax1.legend(fontsize=labelsize,bbox_to_anchor=(-.25,1.2,1.5,0.25), loc=3,ncol=3, mode="expand", borderaxespad=0.,fancybox=True, framealpha=0.5)
    fig.savefig('../figures/hists_flatness_scalexp_{0}.png'.format(name),dpi=150)
    plt.show()  
    
def main_hists_flatness_scalexp__no_compensate():
    from matplotlib import gridspec
    gs=gridspec.GridSpec(3,3)    
    fig = plt.figure(figsize=(10.,10.),facecolor = 'white')
    fig.subplots_adjust(top=0.905,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
    
    plt.figtext(.01,.92,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.92,'b)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.92,'c)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.01,.6,'d)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.6,'e)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.6,'f)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.05,.3,'g)',fontsize=labelsize,fontweight='bold')    
    
    ax0=fig.add_subplot(gs[0,0]) 
    ax1=fig.add_subplot(gs[0,1]) 
    ax2=fig.add_subplot(gs[0,2]) 
    ax3=fig.add_subplot(gs[1,0]) 
    ax4=fig.add_subplot(gs[1,1]) 
    ax5=fig.add_subplot(gs[1,2]) 
    ax6=fig.add_subplot(gs[2,2]) 
    ax7=fig.add_subplot(gs[2:,2])     
    
    to_do=[1,5,13,21,24,29]
    plot_flatness_slope(ax7,range(1,33))
    for i in [21,24]:
        name='_'+str(i)
        alpha=np.loadtxt('parameters/Parameters' +name+'.dat')
        alpha=alpha[2]
        slope,intercept=get_slope(i,normed=True)
        ax7.plot([0.0,alpha,alpha],[slope,slope,-.9],'k--')
    ax7.set_ylim([-0.9,0.0])
    plot_flatness_alpha(ax6,to_do)
    ax6.legend(fontsize=labelsize,loc=1,ncol=2,fancybox=True, framealpha=0.5) 

    for ax,i in zip([ax0,ax1,ax2,ax3,ax4,ax5],to_do):
        plot_Hists(ax,str(i),title=True,Norm=True)
    ax1.legend(fontsize=labelsize,bbox_to_anchor=(-.25,1.2,1.5,0.25), loc=3,ncol=3, mode="expand", borderaxespad=0.,fancybox=True, framealpha=0.5)
    fig.savefig('figures/hists_flatness_scalexp.pdf',dpi=150)
    plt.show()  

    
def main_strucfunc_scalexp():
    from matplotlib import gridspec
    gs=gridspec.GridSpec(3,3)
    
    fig = plt.figure(figsize=(10.,8.),facecolor = 'white')
    fig.subplots_adjust(top=0.905,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
    
    plt.figtext(.01,.92,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.92,'b)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.92,'c)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.01,.6,'d)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.6,'e)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.6,'f)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.05,.3,'g)',fontsize=labelsize,fontweight='bold')    
    
    ax0=fig.add_subplot(gs[0,0]) 
    ax1=fig.add_subplot(gs[0,1]) 
    ax2=fig.add_subplot(gs[0,2]) 
    ax3=fig.add_subplot(gs[1,0]) 
    ax4=fig.add_subplot(gs[1,1]) 
    ax5=fig.add_subplot(gs[1,2]) 
    ax6=fig.add_subplot(gs[2,:]) 
    
    to_do=[1,5,13,21,24,29]
    plot_scaling_exponent(ax6,to_do=to_do)
    ax6.set_ylim([0.0,7.0])
    
    for ax,i in zip([ax0,ax1,ax2,ax3,ax4,ax5],to_do):
        plot_statistics(ax,i,to_do=[2,4,6],title=True)
    ax1.legend(fontsize=labelsize,bbox_to_anchor=(-.25,1.2,1.5,0.25), loc=3,ncol=4, mode="expand", borderaxespad=0.,fancybox=True, framealpha=0.5)
    fig.savefig('figures/strucfunc_scalexp.pdf',dpi=150)
    plt.show() 
    
def main_hists_flatness_logder():
    from matplotlib import gridspec
    gs=gridspec.GridSpec(4,3)
    
    fig = plt.figure(figsize=(10.,10.),facecolor = 'white')
    fig.subplots_adjust(top=0.925,bottom=0.05,left=0.07,right=0.99,hspace=.45,wspace=.31)
    
    plt.figtext(.05,.935,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.38,.935,'b)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.71,.935,'c)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.05,.705,'d)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.38,.705,'e)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.71,.705,'f)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.05,.47,'g)',fontsize=labelsize,fontweight='bold')    
    plt.figtext(.05,.235,'h)',fontsize=labelsize,fontweight='bold')    
    
    ax0=fig.add_subplot(gs[0,0]) 
    ax1=fig.add_subplot(gs[0,1]) 
    ax2=fig.add_subplot(gs[0,2]) 
    ax3=fig.add_subplot(gs[1,0]) 
    ax4=fig.add_subplot(gs[1,1]) 
    ax5=fig.add_subplot(gs[1,2]) 
    ax6=fig.add_subplot(gs[2,:]) 
    ax7=fig.add_subplot(gs[3,:]) 

    to_do=[1,5,13,21,24,29]
    plot_flatness_alpha(ax6,to_do)
    ax6.legend(fontsize=labelsize,loc=1,ncol=2,fancybox=True, framealpha=0.5)
    plot_flatness_derivative(ax7,to_do)
    ax7.legend(fontsize=labelsize,loc=4,ncol=2,fancybox=True, framealpha=0.5)

    for ax,i in zip([ax0,ax1,ax2,ax3,ax4,ax5],to_do):
        plot_Hists(ax,str(i),title=True,Norm=True)
    ax1.legend(fontsize=labelsize,bbox_to_anchor=(-.25,1.2,1.5,0.25), loc=3,ncol=3, mode="expand", borderaxespad=0.,fancybox=True, framealpha=0.5)
    fig.savefig('figures/hists_flatness_logder.pdf',dpi=150)
    plt.show()     
    
def main_strucfunc_scalexp_full(l_name):
    from matplotlib import gridspec
    gs=gridspec.GridSpec(3,3)
    
    fig = plt.figure(figsize=(10.,10.),facecolor = 'white')
    fig.subplots_adjust(top=0.905,bottom=0.06,left=0.080,right=0.99,hspace=.47,wspace=.34)
    
    plt.figtext(.01,.92,'a)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.92,'b)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.92,'c)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.01,.6,'d)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.35,.6,'e)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.69,.6,'f)',fontsize=labelsize,fontweight='bold')
    plt.figtext(.05,.3,'g)',fontsize=labelsize,fontweight='bold')    
    
    ax0=fig.add_subplot(gs[0,0]) 
    ax1=fig.add_subplot(gs[0,1]) 
    ax2=fig.add_subplot(gs[0,2]) 
    ax3=fig.add_subplot(gs[1,0]) 
    ax4=fig.add_subplot(gs[1,1]) 
    ax5=fig.add_subplot(gs[1,2]) 
    ax6=fig.add_subplot(gs[2,:])

    p_to_do=[2,3,4,5,6,7,8]
    to_do=[1,3,5,7,11,13]
    
    plot_scaling_exponent(l_name,ax6,to_do=to_do,normed=False,p_to_do=np.array(p_to_do))
    ax6.legend(fontsize=labelsize,loc=2,ncol=3,fancybox=True, framealpha=0.5)  
    ax6.set_ylim([0.0,5.0])
    
    for ax,i in zip([ax0,ax1,ax2,ax3,ax4,ax5],to_do):
        plot_statistics(l_name,ax,i,to_do=p_to_do,title=True,normed=True,compensated=False,scale=30)
        ax.set_ylim([10**-7,150])
        #ax.set_xlim([2.0**-8,0.1])
    ax1.legend(fontsize=labelsize,bbox_to_anchor=(-1.0,1.2,3.2,0.25), loc=3,ncol=7, mode="expand", borderaxespad=0.,fancybox=True, framealpha=0.5)
    fig.savefig('../figures/strucfunc_scalexp_full_{0}.png'.format(l_name),dpi=150)
    plt.show()
    
def init_parallel(i):
    print('Doing '+str(i))
    plot_statistics(i)
    plot_skewness(i)
    plot_Hists(i)
    plot_flatness(i)

if __name__=='__main__':
    main_strucfunc_scalexp_full('N15_k0')
    main_strucfunc_scalexp_full('N15_k5')
    main_strucfunc_scalexp_full('N15_k7')
    main_hists_flatness_scalexp('N15_k0',[1,3,7,11,13,15])
    main_hists_flatness_scalexp('N15_k5',[1,3,7,11,13,15])
    main_hists_flatness_scalexp('N15_k7',[1,3,7,11,13,15])