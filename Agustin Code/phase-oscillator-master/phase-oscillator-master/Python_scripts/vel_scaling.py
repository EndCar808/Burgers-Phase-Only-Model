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
from plot_miguel import load_parameters, u_rms, log_derivative
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation

animation.writers['ffmpeg'](
            fps = 30,
            metadata = {'title': ''},
            codec = 'libx264',
            extra_args = [#'-preset', 'slow',
                          '-pix_fmt', 'yuv420p',
                          '-crf', '19',
                          '-profile:v', 'high',
                          '-threads', '8']) 

def plot_struc_func(ax,
                x,
                IS,
                urms,
                moments_to_plot=[2,4],
                labels=[r'$S_2(r)$',r'$S_4(r)$'],
                style=['r','g'],
                ):
    ims=[]
    for moment,label,line in zip(moments_to_plot,labels,style):
        im,=ax.plot(x,np.abs(IS[moment-1])/urms**moment,line,label=label)
        ims.append(im)
    return ims
        
def plot_zeta_p(ax,
                x,
                IS,
                urms,
                moments_to_plot=[2,4],
                labels=[r'$S_2(r)$',r'$S_4(r)$'],
                style=['r','g'],
                ):
    ims=[]
    for moment,label,line in zip(moments_to_plot,labels,style):
        px,py=log_derivative(x,np.abs(IS[moment-1])/urms**moment)
        im0,=ax.plot(px,py,line,label=label)
        x0_py=(py[1]-py[0])/(px[1]-px[0])*(x[0]-px[0])+py[0]
        im1,=ax.plot([x[0],px[0]],[x0_py,py[0]],'k--')
        ims.append(im0)
        ims.append(im1)
    return ims

def plot_RS(ax,RealSpace,N,x):
    try:
        RS=RealSpace.read_reals(kr).reshape((2*N), order="F")
    except:
        return 0,1
    im,=ax.plot(x,RS,'b')
    return im,0

def main(base_dir='',name_list=[1]):
    moments_to_plot=[2,3,4,5,6]
    labels=[r'$p=2$',r'$p=3$',r'$p=4$',r'$p=5$',r'$p=6$']
    style=['r','g','b','y','c']
    style2=['r--','g--','b--','y--','c--']
    
    from matplotlib import gridspec
    gs=gridspec.GridSpec(2,1)
    
    fig = plt.figure(figsize=(6.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.95,bottom=0.08,left=0.11,right=0.98,hspace=0.22)
    N,k0,alpha,beta,n_max,steps,transient,binning,file_name=load_parameters(base_dir='',name=str(name_list[0]))
    fig.suptitle(r'$N={0}, k_0={1}, \alpha={2}, \beta={3}$'.format(N,k0,alpha,beta),fontsize=labelsize)
    ax0  = fig.add_subplot(gs[0,0])
    ax1  = fig.add_subplot(gs[1,0])
    ax0.tick_params(labelsize=ticksize)
    ax1.tick_params(labelsize=ticksize) 
    
    urms=u_rms(N=N,k0=k0,alpha=alpha,beta=beta)
    name=name_list[0]
    f_x=FortranFile('realspace/RealSpaceCoordinates_{0}.dat'.format(name), 'r')
    x=np.zeros(2*N,dtype=kr)
    for i in range(2*N):
        x[i]=f_x.read_reals(kr)
    f_x.close()
    del f_x
    int_len=np.pi/(k0+1.0)
    x_IS=x[1:N+1]/int_len
    
    ax0.set_xlim([0.0,2.0*np.pi])
    ax0.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
    ax0.set_xticklabels(["0","$\pi/2$","$\pi$","$3\pi/2$","$2\pi$"])
    ax0.set_ylabel(r"$u(t,x)/u_\text{rms}$", fontsize=labelsize)
    ax0.set_xlabel(r'$x$',fontsize=labelsize)

    ax1.set_ylabel(r'$\vert S_p(r)\vert/{u_\text{rms}}^p$',fontsize=labelsize)
    ax1.set_xlabel(r'$r/L$',fontsize=labelsize)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim([x_IS[0],1.0])
    ax1.set_ylim([10**-4,50])
                
    axins = inset_axes(ax1, width=1.6, height=1.3,loc=4)#10,bbox_to_anchor=[390.,105])
    axins.set_ylabel(r'$\zeta_p$',fontsize=labelsize)
    
    axins.plot([x[0],1.0],[1.0,1.0],'k--')
    axins.set_xscale('log')
    axins.set_xlim([x_IS[0],1.0])
    axins.set_ylim([0,2])#moments_to_plot[-1]])
    axins.tick_params(labelsize=ticksize-5)

    #ax1.legend(fontsize=labelsize,loc=2,ncol=1,fancybox=True, framealpha=0.5)

    ims=[]  
    window_size=5
    IS_c=np.zeros((n_max,N),dtype='float64')
    count=0

    for name in name_list:    
        f=FortranFile(base_dir+'statistics/IS_{0}.dat'.format(name), 'r')
        RealSpace=FortranFile('realspace/RealSpace_{0}.dat'.format(name), 'r')
        while True:
            try:
                IS=f.read_reals(kr).reshape((n_max,N), order="F")
        #            if count == 0 :
        #                IS_c[:]=IS
        #            else:
        #                IS_c[count%window_size]=IS
                IS_c+=IS
                count+=1
            except:
                break
            im0,status1=plot_RS(ax0,RealSpace,N,x)
            if status1==1:
                break
            else:
                im1=plot_struc_func(ax1,
                                    x_IS,
                                    IS,
                                    urms,
                                    moments_to_plot=moments_to_plot,
                                    labels=labels,
                                    style=style
                                    )
                im2=plot_zeta_p(axins,
                                x_IS,
                                IS,
                                urms,
                                moments_to_plot=moments_to_plot,
                                labels=labels,
                                style=style
                                )
                im3=plot_struc_func(ax1,
                                    x_IS,
                                    IS_c/float(count),#np.average(IS_c,axis=0),
                                    urms,
                                    moments_to_plot=moments_to_plot,
                                    labels=labels,
                                    style=style2
                                    )
                im4=plot_zeta_p(axins,
                                x_IS,
                                IS_c/float(count),#np.average(IS_c,axis=0),
                                urms,
                                moments_to_plot=moments_to_plot,
                                labels=labels,
                                style=style2
                                )
#                iml=[]
#                iml.append(im0)
#                for i in [im1,im2,im3,im4]:
#                    for j in i:
#                        iml.append(j)
#                ims.append(iml)
                ims.append([im0]+im1+im2+im3+im4)
        RealSpace.close
        f.close
        del RealSpace, f
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save('vel_scaling_N_{0}.mp4'.format(int(np.log2(N))), dpi=180)
    plt.show()
    return 0

if __name__ == '__main__':
    from datetime import datetime
    ta=datetime.now() 
    main(name_list=[5,6,7,8])
    tb=datetime.now()
    print("Done. It took "+str(tb-ta)+".")
