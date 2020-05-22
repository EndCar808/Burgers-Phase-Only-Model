import matplotlib
matplotlib.use('Agg') 
from plot_routines import plot_Hists, plot_transition, plot_statistics,plot_flatness_alpha, load_statistics, plot_transition
import matplotlib.pyplot as plt

for k in [0,5,7]:

    name='N15_k{0}'.format(k)
    
    plot_transition(name,range(1,17),0,pre_name=name)
    
    fig0 = plt.figure(figsize=(10.,10.),facecolor = 'white')
    fig0.subplots_adjust(top=0.96,bottom=0.05,left=0.07,right=0.99)
    ax0=fig0.add_subplot(111)
    
    moment=0
    import numpy as np
    flatness_0=np.empty(16)
    
    for i in range(1,17):
        flatness_0[i-1]=plot_Hists(ax0,name+'_'+str(i),stat_var='all',title=False,Norm=True,moment=moment)
    ax0.set_xscale('log')
    ax0.set_xlim([0.18,10.0**2])
    ax0.set_ylim([10.0**-8,10.0**0])
    
    fig0.suptitle('Oscillators premultiplied PDFs. '+r'$N=2^{15}$, $\log_2 k_0=$'+str(k)+', moment='+str(moment),fontsize=15)
    fig0.savefig('../figures/premult_pdf_{0}_{1}.png'.format(name,moment),dpi=150)
    
    alpha_list,skewness,flatness=load_statistics(name,range(1,17),0)
    
    for a,i,j in zip(alpha_list,flatness,flatness_0):
        print(a,i,j)

#fig1 = plt.figure(figsize=(10.,10.),facecolor = 'white')
#fig1.subplots_adjust(top=0.905,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
#ax1=fig1.add_subplot(111)
#for i in range(1,17):
#    plot_statistics('N15_k1',ax1,i)
#
#fig2 = plt.figure(figsize=(10.,10.),facecolor = 'white')
#fig2.subplots_adjust(top=0.905,bottom=0.06,left=0.07,right=0.99,hspace=.47,wspace=.34)
#ax2=fig2.add_subplot(111)
#plot_flatness_alpha('N15_k1',ax2,range(1,17))