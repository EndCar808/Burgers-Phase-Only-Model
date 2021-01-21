#Agustin Arguedas

import numpy as np
if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    from plot_routines import labelsize, ticksize
else:
    labelsize=8
    ticksize=6

from utilities import load_parameters, load_statistics, load_Hists

def plot_percentiles(ax,to_load=[0],r_index=0,title='',name='',percentile_limit=5.0,tail='right',base_dir='../',legend=True,percentile_to_load = 'du'):
    #Formatting 
    ax.tick_params(labelsize=ticksize)
    ax.set_xlabel(r'$\alpha$',fontsize=labelsize)
    ax.set_title(r'Normalized percentile outside ${0}\:\sigma$'.format(percentile_limit),fontsize=labelsize)
    #Generate Gaussian distribution to normalize percentile
    x=np.linspace(-200,200,num=500001)
    Gauss=np.exp(-x**2*0.5)/np.sqrt(2.0*np.pi)
    dx=x[1]-x[0]
    if np.sum(Gauss*dx) < 0.99 :
        print('Gauss not normed! It gives {0}.'.format(np.sum(Gauss*dx)))
    #Generate histograms for both, left, and right tails
    Gauss_b = np.where(np.abs(x)>=percentile_limit,Gauss,0.0)
    Gauss_l = np.where(x<=-percentile_limit,Gauss,0.0)
    Gauss_r = np.where(x>= percentile_limit,Gauss,0.0)
    #Riemann sum to approximate Gauss integral
    Gauss_percentile_b = np.sum(Gauss_b*dx)
    Gauss_percentile_l = np.sum(Gauss_l*dx)
    Gauss_percentile_r = np.sum(Gauss_r*dx)
    #Clear up memory
    del x, Gauss, dx
    #Generate arrays to save data percentiles
    pars=load_parameters(name=name+'_{0}'.format(to_load[0]),base_dir=base_dir)
    percentiles_b=np.empty(len(to_load),dtype='float64')
    percentiles_l=np.empty(len(to_load),dtype='float64')
    percentiles_r=np.empty(len(to_load),dtype='float64')
    alphas=np.empty(len(to_load),dtype='float64')
    control=0
    for i in to_load:
        #Load histograms from data
        pars=load_parameters(name=name+'_{0}'.format(i),base_dir=base_dir)
        H_du, bins_du, dx_du = load_Hists(name+'_{0}'.format(i),percentile_to_load,Norm=True,reduction=False,base_dir=base_dir)
        if np.sum(H_du*dx_du) < 0.98 :
            print('H not normed! It gives {0}.'.format(np.sum(H_du*dx_du)))
        #Prepare data histograms for both, left, and right tail
        H_b = np.where(np.abs(bins_du)>=percentile_limit,H_du,0.0)
        H_l = np.where(bins_du<=-percentile_limit,H_du,0.0)
        H_r = np.where(bins_du>=percentile_limit,H_du,0.0)
        #Get the percentile from from Riemann sum
        percentiles_b[control]=np.sum(H_b*dx_du)
        percentiles_l[control]=np.sum(H_l*dx_du)
        percentiles_r[control]=np.sum(H_r*dx_du)
        #Load aplha
        alphas[control]=pars['alpha']
        control+=1
    #Plot left and right tails
    p1 = ax.plot(alphas,percentiles_l/Gauss_percentile_l,c='C2',label='Left tail')
    p2 = ax.plot(alphas,percentiles_r/Gauss_percentile_r,c='C3',label='Right tail')
    ax.set_xlim([alphas[0],alphas[-1]])
    #Formatting and legend
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if __name__ == '__main__':
        axins = ax.inset_axes( [0.19,8500,0.91,10000], transform=ax.transData)#width=1.3, height=0.9,loc=2)
        axins.set_xlim([alphas[0],alphas[-1]])
    else:
        axins = ax.inset_axes( [0.22,8500,0.95,10000], transform=ax.transData)#width=1.3, height=0.9,loc=2)
        axins.set_xlim([0.0,2.5])
        axins.set_xticks([0,0.5,1.0,1.5,2.0])
    axins.tick_params(labelsize=ticksize)
    ax.set_xticks([0,.5,1,1.5,2.0,2.5,3.0,3.5])
    axins.set_yscale('log')
    axins.set_yticks([0.1,1.0,10,100.0,1000.0,10000])
    axins.set_ylim([0.09,25000])
    axins.set_yticklabels(['',r'$10^0$','',r'$10^2$','',r'$10^4$'])
    axins.set_xlabel(r'$\alpha$',fontsize=ticksize)
    axins.plot(alphas,percentiles_l/Gauss_percentile_l,c=p1[0].get_color())
    axins.plot(alphas,percentiles_r/Gauss_percentile_r,c=p2[0].get_color())
    
    if legend == True:
        ax.legend(fontsize=labelsize,loc=0,fancybox=True, framealpha=0.5, bbox_to_anchor=(0.9, -0.2), ncol = 2)

def main_percentiles(to_load,r_index,title,name,ax=None,fig=None,save=False,percentile_limit=5.0,base_dir='../',legend=True,fig_path=None,percentile_to_load = 'du'):
    #Generate figure, if not given as parameter
    if ax == None:
        fig = plt.figure(figsize=(5.,3.),facecolor = 'white')
        fig.subplots_adjust(top=0.90,bottom=0.29,left=0.110,right=0.97)
        ax  = fig.add_subplot(111)
    #Call plot routine
    plot_percentiles(ax,to_load=to_load,r_index=r_index,title=title,name=name,percentile_limit=percentile_limit,base_dir=base_dir,legend=legend,percentile_to_load = percentile_to_load)
    #Save figure
    if save==True:
        if fig_path == None:
            fig_path = '../figures/main_percentiles/plot_percentiles_{0}_{1}_{2}.png'.format(name,percentile_limit,percentile_to_load)
            fig_path = '../figures/main_percentiles/plot_percentiles_{0}_{1}_{2}.pdf'.format(name,percentile_limit,percentile_to_load)
        fig.savefig(fig_path,dpi=320)
        plt.show()
    return fig,ax

if __name__=='__main__':
    to_load=range(1,289)
    name='N13_k0'
    title=r'$N=2^{13}, k_0=1$.'
    r_index=0
    fig,ax=main_percentiles(to_load,r_index,title,name,save=True,percentile_limit=5.0,percentile_to_load = 'du')