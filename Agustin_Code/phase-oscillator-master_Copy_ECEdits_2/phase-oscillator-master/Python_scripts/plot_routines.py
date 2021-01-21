#Agustin Arguedas

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
labelsize=15
ticksize=12

from utilities import load_Hists, load_parameters, load_RS, load_stats, load_statistics
from utilities import get_log_derivative, get_slope, get_slope_list

def log_log_Hists(ax,
                  name,
                  stat_var='all',
                  title=False,
                  Norm=False,
                  moment=0,
                  base_dir='../'
                  ):
    pars=load_parameters(name=name,base_dir=base_dir)
    labels={'u':r'$r=\pi$','du':r'$r=\pi/N$','Grad':r'$\partial_x u$'}
    if stat_var=='all':
        H_u, bins_u , dx_u= load_Hists(name,'u',Norm=Norm)
        H_du, bins_du, dx_du = load_Hists(name,'du',Norm=Norm)
        H_Grad, bins_Grad, dx_Grad = load_Hists(name,'Grad',Norm=Norm)
        H_u   =H_u   *np.abs(bins_u   )**moment
        H_du  =H_du  *np.abs(bins_du  )**moment
        H_Grad=H_Grad*np.abs(bins_Grad)**moment
        ax.plot(bins_u ,H_u ,label=r'$r=\pi$')
        ax.plot(bins_du,H_du,label=r'$r=\pi/N$')
        ax.plot(bins_Grad,H_Grad,label=r'$\partial_x u$')
        ax.set_yscale('log')
        ax.tick_params(labelsize=ticksize)
        xmin=np.min([bins_u.min(),bins_du.min(),bins_Grad.min()])
        xmax=np.max([bins_u.max(),bins_du.max(),bins_Grad.max()])
        ymin=np.min([H_u.min(),H_du.min(),H_Grad.min()])
        ymax=np.max([H_u.max(),H_du.max(),H_Grad.max()])
    else:
        H, bins , dx= load_Hists(name,stat_var,Norm=Norm)
        H=H*np.abs(bins)**moment
        ax.plot(bins ,H ,label=labels[stat_var])
        ax.set_yscale('log')
        ax.tick_params(labelsize=ticksize)
        xmin=bins.min()
        xmax=bins.max()
        ymin=H.min()
        ymax=H.max()
    ax.set_xlim([xmin,xmax])
    ax.set_xlim([-40,40])
    ax.set_ylim([ymin*0.5,ymax*2.0])
    if Norm==False:
        ax.set_xlabel(r'$\delta u_r$',fontsize=labelsize)
    else:
        ax.set_xlabel(r'$\delta u_r/\sigma$',fontsize=labelsize)
        binsX=np.linspace(xmin,xmax,num=251)
        Gauss=np.exp(-binsX**2/2)/np.sqrt(2*np.pi)
        ax.plot(binsX,Gauss,'k--')
    ax.set_ylabel('PDF',fontsize=labelsize)
    if title==True:
        ax.set_title(r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)), fontsize=labelsize)
    return np.sum(H_du*dx_du)

def plot_alpha_scaling_exponent(l_name,
                                ax,
                                to_do,
                                normed=True,
                                p_to_do=np.array([4]),
                                scale=5,
                                base_dir='../'
                                ):
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel(r'$\alpha$',fontsize=labelsize)
    ax.set_ylabel('Flatness slope',fontsize=labelsize)
    alpha_list=np.zeros(len(to_do))
    scaling_exponent=np.zeros(len(to_do))
    control=0
    for i in to_do:
        name=l_name+'_'+str(i)
        pars=load_parameters(name=name,base_dir=base_dir)
        alpha_list[control]=pars['alpha']
        scaling_exponent[control]=get_scaling_exponent_list(i,moment=p_to_do,normed=normed,scale=scale)
        control+=1
    ax.plot(alpha_list,scaling_exponent)
    ax.set_xlim([alpha_list[0],alpha_list[-1]])
    
def plot_flatness(l_name,
                  i,
                  base_dir='../'
                  ):
    name=l_name+'_'+str(i)
    pars=load_parameters(name=name,base_dir=base_dir)
    x=load_RS(name,'RealSpaceCoordinates')
    x=x[1:pars['N']+1]
    fig = plt.figure(figsize=(12.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.12,left=0.10,right=0.79)
    fig.suptitle(r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([x[0],x[-1]])
    ax.set_xlabel(r'$r$',fontsize=labelsize)
    ax.set_ylabel('Structure functions',fontsize=labelsize)
    subname='increment_statistics'
    IS=load_stats(name,subname)
    IS=IS.reshape(pars['N'],pars['n_max']).T
    ax.plot(x,np.abs(IS[3])/np.abs(IS[1])**2)
    #ax.legend(fontsize=labelsize,loc=1,bbox_to_anchor=(1.32,1.))
    #ax.set_ylim([np.min(np.abs(IS[-1,0]))*0.1,np.max(np.abs(IS))*10])
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.savefig('figures/flatness'+name+'.pdf',dpi=150)
    #plt.show()
    plt.close(fig)

def plot_flatness_alpha(l_name,
                        ax,
                        to_do,
                        scaling=True,
                        scale=10,
                        compensated=False,
                        base_dir='../'
                        ):
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    if compensated==False:
        ax.set_ylabel('Velocity increment flatness',fontsize=labelsize)
        ax.set_ylabel(r'Flatness $S_4(r)/S_2(r)^2$',fontsize=labelsize)
    else:
        ax.set_ylabel('Compensated flatness',fontsize=labelsize)
    vmin=0.9
    vmax=1.0
    i = to_do[0]
    name=l_name+'_'+str(i)
    pars=load_parameters(name=name,base_dir=base_dir)
    int_len=2.0*np.pi/pars['k0']
    for i in to_do:
        name=l_name+'_'+str(i)
        pars=load_parameters(name=name,base_dir=base_dir)
        x=load_RS(name,'RealSpaceCoordinates')
        x=x[1:pars['N']+1]/int_len
        subname='increment_statistics'
        IS=load_stats(name,subname)
        IS=IS.reshape(pars['N'],pars['n_max']).T
        flatness=np.abs(IS[3])/np.abs(IS[1])**2
        if compensated==True:
            m,b=get_slope(l_name,i,moment=4,normed=True,scale=scale)#This has to be taken out
            flatness/=x**m*10.0**(b)
            x_scale=x[1:scale]
            ax.plot(x_scale,np.ones(x_scale.size),'k--')
        ax.plot(x,flatness,label=r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)))
        vmin=np.min([vmin,np.min(flatness)])
        vmax=np.max([vmax,np.max(flatness)])
        if scaling==True and compensated==False:#This has to be taken out
            m,b=get_slope(l_name,i,moment=4,normed=True,scale=scale)
            x_scale=x[1:scale]
            ax.plot(x_scale,x_scale**m*10.0**(b),'k--')
    #ax.plot([x[0]/int_len,x[-1]/int_len],[3.0,3.0],'k--')
    #ax.plot([int_len,int_len],[vmin,vmax],'k--')
    ax.set_ylim([vmin,vmax])
    ax.set_xlim([x[0],x[-1]])
    ax.set_yscale('log')
    ax.set_xscale('log')

def plot_flatness_derivative(l_name,
                             ax,
                             to_do=[1],
                             base_dir='../'
                             ):
    for i in to_do:
        name=l_name+'_'+str(i)
        pars=load_parameters(name=name,base_dir=base_dir)
        x,derivative=get_log_derivative(l_name,i,moment=4,normed=True)
        ax.plot(x,derivative,label=r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)))
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    ax.set_ylabel(r'$\mathrm{d}\log\mathrm{Flatness}/\mathrm{d}\log r$',fontsize=labelsize)

def plot_flatness_slope(l_name,
                        ax,
                        to_do,
                        normed=True,
                        p_to_do=np.array([4]),
                        scale=10,
                        base_dir='../'
                        ):
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel(r'$\alpha$',fontsize=labelsize)
    ax.set_ylabel('Flatness slope',fontsize=labelsize)
    alpha_list=np.zeros(len(to_do))
    slope=np.zeros(len(to_do))
    control=0
    for i in to_do:
        name=l_name+'_'+str(i)
        pars=load_parameters(name=name,base_dir=base_dir)
        alpha_list[control]=pars['alpha']
        x,sl=get_log_derivative(l_name,
                       i=i,
                       moment=4,#Set this here by hand!
                       normed=True,
                       base_dir='../'
                       )#get_slope(l_name,i,moment=moment[j],normed=normed,scale=scale,base_dir=base_dir)
        slope[control]=sl[16]
#        slope[control],intercept[control]=get_slope(l_name,i,moment=4,normed=normed,scale=scale)
        control+=1
    ax.plot(alpha_list,slope)
    ax.set_xlim([alpha_list[0],alpha_list[-1]])

def plot_Hists(ax,
               name,
               stat_var='all',
               title=False,
               Norm=False,
               moment=0,
               base_dir='../'
               ):
    pars=load_parameters(name=name,base_dir=base_dir)
    labels={'u':r'$r=\pi$','du':r'$r=\pi/N$','Grad':r'$\partial_x u$'}
    if stat_var=='all':
        H_u, bins_u , dx_u= load_Hists(name,'u',Norm=Norm)
        H_du, bins_du, dx_du = load_Hists(name,'du',Norm=Norm)
        H_Grad, bins_Grad, dx_Grad = load_Hists(name,'Grad',Norm=Norm)
        H_u   =H_u   *np.abs(bins_u   )**moment
        H_du  =H_du  *np.abs(bins_du  )**moment
        H_Grad=H_Grad*np.abs(bins_Grad)**moment
        ax.plot(bins_u ,H_u ,label=r'$r=\pi$')
        ax.plot(bins_du,H_du,label=r'$r=\pi/N$')
        ax.plot(bins_Grad,H_Grad,label=r'$\partial_x u$')
        ax.set_yscale('log')
        ax.tick_params(labelsize=ticksize)
        xmin=np.min([bins_u.min(),bins_du.min(),bins_Grad.min()])
        xmax=np.max([bins_u.max(),bins_du.max(),bins_Grad.max()])
        ymin=np.min([H_u.min(),H_du.min(),H_Grad.min()])
        ymax=np.max([H_u.max(),H_du.max(),H_Grad.max()])
    else:
        H, bins , dx= load_Hists(name,stat_var,Norm=Norm)
        H=H*np.abs(bins)**moment
        ax.plot(bins ,H ,label=labels[stat_var])
        ax.set_yscale('log')
        ax.tick_params(labelsize=ticksize)
        xmin=bins.min()
        xmax=bins.max()
        ymin=H.min()
        ymax=H.max()
    ax.set_xlim([xmin,xmax])
    #ax.set_xlim([-40,40])
    ax.set_ylim([ymin*0.5,ymax*2.0])
    if Norm==False:
        ax.set_xlabel(r'$\delta u_r$',fontsize=labelsize)
    else:
        ax.set_xlabel(r'$\delta u_r/\sigma$',fontsize=labelsize)
        binsX=np.linspace(xmin,xmax,num=251)
        Gauss=np.exp(-binsX**2/2)/np.sqrt(2*np.pi)
        ax.plot(binsX,Gauss,'k--')
    ax.set_ylabel('PDF',fontsize=labelsize)
    if title==True:
        ax.set_title(r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)), fontsize=labelsize)
    return np.sum(H_du*dx_du)

def plot_statistics(l_name,
                    ax,i,
                    to_do=[2,3,4],
                    normed=True,
                    title=False,
                    scale=10,
                    compensated=True,
                    base_dir='../'
                    ):
    from utilities import u_rms
    name=l_name+'_'+str(i)
    pars=load_parameters(name=name,base_dir=base_dir)
    subname='control_statistics'
    CS=load_stats(name,subname)
    labels=[r'$\langle\delta_ru\rangle$',r'$\langle\delta_ru^2\rangle$',r'$\langle\delta_ru^3\rangle$',r'$\langle\delta_ru^4\rangle$',r'$\langle\delta_ru^5\rangle$',r'$\langle\delta_ru^6\rangle$',r'$\langle\delta_ru^7\rangle$',r'$\langle\delta_ru^8\rangle$']
    labels=[r'$p=1$',r'$p=2$',r'$p=3$',r'$p=4$',r'$p=5$',r'$p=6$',r'$p=7$',r'$p=8$']
    x=load_RS(name,'RealSpaceCoordinates')
    int_len=2.0*np.pi/pars['k0']
    x=x[1:pars['N']+1]/int_len
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([x[0],x[-1]])
#    ax.set_xticks([0,0.25*np.pi,0.5*np.pi,0.75*np.pi,np.pi])
#    ax.set_xticklabels(["0","$\pi/4$","$\pi/2$","$3\pi/4$","$\pi$"])
    ax.set_xlabel(r'$r/L$',fontsize=labelsize)
    ax.set_ylabel('Structure functions',fontsize=labelsize)
    ax.set_ylabel(r'$S_p(r)/{u_\textrm{rms}}^p$',fontsize=labelsize)
    subname='increment_statistics'
    IS=load_stats(name,subname)
    IS=IS.reshape(pars['N'],pars['n_max']).T
    urms=u_rms(N=pars['N'],k0=pars['k0'],alpha=pars['alpha'],beta=pars['beta'])#np.sqrt(np.abs(IS[1]))
    if normed==False and compensated==False:
         ax.plot(x,CS,'k--',label='Control')
         ax.plot(x[16],1.0,'ro')
         for j in to_do:
             ax.plot(x,np.abs(IS[j-1]),label=labels[j-1])
    elif normed==False and compensated==True:#Take these compensated out
         for j in to_do:
             m,b=get_slope(l_name,i,moment=j,normed=False,scale=scale)
             ax.plot(x,np.abs(IS[j-1])/(x**m*10.0**(b)),label=labels[j-1])
         x_scale=x[1:scale]
         ax.plot(x_scale,np.ones(x_scale.size),'k--')
    elif normed==True and compensated==False:
         for j in to_do:
             ax.plot(x,np.abs(IS[j-1])/urms**j,label=labels[j-1])
    elif normed==True and compensated==True:#Take these compensated out
         for j in to_do:
             m,b=get_slope(l_name,i,moment=j,normed=True,scale=scale)
             ax.plot(x,np.abs(IS[j-1])/urms**j/(x**m*10.0**(b)),label=labels[j-1])
         x_scale=x[1:scale]
         ax.plot(x_scale,np.ones(x_scale.size),'k--')
    else:
        print('False normed option.')
        return 1
    ax.plot([x[16],x[16]],[10**-7,150],'k--')
    ax.set_yscale('log')
    ax.set_xscale('log')
    if title==True:
        ax.set_title(r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)), fontsize=labelsize)

def plot_scaling_exponent(l_name,
                          ax,
                          to_do,
                          normed=False,
                          p_to_do=np.arange(2,9),
                          base_dir='../'
                          ):
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel(r'$p$',fontsize=labelsize)
    ax.set_ylabel('Scaling exponent\n of '+r'$S_p(r)/{u_\mathrm{rms}}^p$',fontsize=labelsize)
    for i in to_do:
        name=l_name+'_'+str(i)
        pars=load_parameters(name=name,base_dir=base_dir)
        slope_list=get_slope_list(l_name,i,moment=p_to_do,normed=normed)
        ax.plot(p_to_do,slope_list,label=r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)))

def plot_skewness(l_name,
                  i,
                  base_dir='../'
                  ):
    name=l_name+'_'+str(i)
    pars=load_parameters(name=name,base_dir=base_dir)
    x=load_RS(name,'RealSpaceCoordinates')
    subname='increment_statistics'
    IS=load_stats(name,subname)
    IS=IS.reshape(pars['N'],pars['n_max']).T

    fig = plt.figure(figsize=(9.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.92,bottom=0.12,left=0.14,right=0.97)
    fig.suptitle(r'$\alpha=$'+str(np.around(pars['alpha'],decimals=5)),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([0.0,np.pi])
    ax.set_xticks([0,0.25*np.pi,0.5*np.pi,0.75*np.pi,np.pi])
    ax.set_xticklabels(["0","$\pi/4$","$\pi/2$","$3\pi/4$","$\pi$"])
    ax.set_xlabel(r'$r$',fontsize=labelsize)
    ax.set_ylabel('Skewness',fontsize=labelsize)
    ax.plot(x,IS[2]/IS[1]**1.5)
    ax.plot(x,x*0.0,'k--')
    fig.savefig('figures/skewness'+name+'.pdf',dpi=150)
    #plt.show()
    plt.close(fig)

def plot_transition(l_name,
                    to_load,
                    r_index,
                    pre_name='',
                    do_skewness=True,
                    do_flatness=True,
                    ax=None,
                    skewness_label='Skewness',
                    flatness_label='Flatness',
                    refs=True,
                    do_legend=True,
                    base_dir='../',
                    linestyle='-',
                    color='k'
                    ):
    name=l_name+'_'+str(to_load[0])
    x=load_RS(name,'RealSpaceCoordinates',base_dir=base_dir)
    pars=load_parameters(name=name,base_dir=base_dir)
    x=x[1:pars['N']+1]
    r=x[r_index]
    del x,name,pars['N']

    alpha_list,skewness,flatness=load_statistics(l_name,to_load,r_index,base_dir=base_dir)

    if ax==None:
        fig = plt.figure(figsize=(10.,4.),facecolor = 'white')
        fig.subplots_adjust(top=0.90,bottom=0.12,left=0.07,right=0.98)
        fig.suptitle('Moments of the velocity increments at '+r'$r=$'+str(np.around(r,decimals=5)),fontsize=labelsize)
        ax  = fig.add_subplot(111)
        ax.tick_params(labelsize=ticksize)
        ax.set_xlabel(r'$\alpha$',fontsize=labelsize)
    if do_skewness==True:
        ax.plot(alpha_list,skewness,label=skewness_label,linestyle=linestyle,color=color)
        if refs == True:
            ax.plot(alpha_list,np.zeros(len(to_load)),'k--')
    if do_flatness==True:
        ax.plot(alpha_list,flatness,label=flatness_label,linestyle=linestyle,color=color)
        if refs == True:
            ax.plot(alpha_list,3.0*np.zeros(len(to_load)),'k--')
    if do_legend==True:
        ax.legend(fontsize=labelsize,loc=0,fancybox=True, framealpha=0.5)
    if ax==None:
        fig.savefig('../figures/{0}transition_{1}.png'.format(pre_name,r_index),dpi=150)
        plt.show()
        plt.close(fig)