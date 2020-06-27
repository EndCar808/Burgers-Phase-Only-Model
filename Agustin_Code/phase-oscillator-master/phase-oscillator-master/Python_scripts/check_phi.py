#Agustin Arguedas

from scipy.io import FortranFile
kr='float64'

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

import matplotlib.animation as animation

animation.writers['ffmpeg'](
    fps=30,
    metadata={'title': ''},
    codec='libx264',
    extra_args=[  # '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        '-crf', '19',
        '-profile:v', 'high'])

def load_phi(base_dir='',name=1):
    N,k0,alpha,beta=load_parameters(base_dir='',name=str(name))
    f=FortranFile(base_dir+'instances/end_phi_{0}.dat'.format(name), 'r')
    phi=f.read_reals(kr).reshape(N, order="F") 
    f.close()
    return phi
        
def plot_phi(ax,
             base_dir='',
             name=1
                ):
    N,k0,alpha,beta=load_parameters(base_dir='',name=str(name))
    phi=load_phi(base_dir=base_dir,name=name)#np.mod(load_phi(base_dir=base_dir,name=name),2.0*np.pi)
    dx=(np.mod((phi-np.pi*0.5)/np.arange(1,N+1),2.0*np.pi))
    dx=np.average(dx[N//2:])
    k_list,p_list=TriadList(N,k0)
    triads=np.mod(phi[p_list-1]+phi[k_list-p_list-1]-phi[k_list-1],2.0*np.pi)
    ax.plot(np.arange(1,triads.size+1),triads-np.pi*0.5)
    print('Look! Triads are pi/2 in the shock fixed point!!')
    #phi=phi+dx*np.arange(1,N+1)    
    #phi=np.mod(phi,2.0*np.pi)
    #ax.plot(np.arange(1,N+1),phi,'.')
    #ax.set_xlim([k0+1,N])
    #ax.set_ylim([0.0,2.0*np.pi])
    #ax.set_yticks([0.0,np.pi*0.5,np.pi,1.5*np.pi,2.0*np.pi])
    #ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])

    ax.set_yticks([0.0,np.pi*0.5,np.pi,1.5*np.pi,2.0*np.pi])
    ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
    ax.set_yscale('log')
    ax.set_xscale('log')
    
def load_triads(f,N,k_list,p_list):
    try:
        phi=f.read_reals(kr).reshape(N, order="F")
    except:
        return 0,1
    triads=np.zeros([N//2+1,N+1],dtype='float64')
    triads[p_list,k_list]=np.mod(phi[p_list-1]+phi[k_list-p_list-1]-phi[k_list-1],2.0*np.pi)
    triads=np.where(triads==0.0,np.nan,triads)
    triads=np.flipud(triads)

    return triads,0

def plot_triange(ax,f,N,k_list,p_list):
    triads,status=load_triads(f,N,k_list,p_list)
    if status==1:
        return 0,1
    im=ax.imshow(triads,interpolation='none',extent=[0,N,0,N/2],vmin=0.0,vmax=2.0*np.pi,cmap='rainbow')
    return im,0
    
def plot_RS(ax,RealSpace,N,x):
    try:
        RS=RealSpace.read_reals(kr).reshape((2*N), order="F")
    except:
        return 0,1
    im,=ax.plot(x,RS,'b')
    return im,0
    
def TriadList(N,k0):
    total_triads=N**2//4-N*k0+k0**2
    l=0
    k_list=np.empty(total_triads,dtype=np.int)
    p_list=np.empty(total_triads,dtype=np.int)
    for p in range(k0+1,N//2+1):
        for k in range(2*p,N+1):
            k_list[l]=k
            p_list[l]=p
            l=l+1
    return k_list,p_list
    
def main(base_dir='',name=1):
    from matplotlib import gridspec
    gs=gridspec.GridSpec(2,1)
    
    fig = plt.figure(figsize=(6.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.95,bottom=0.06,left=0.11,right=0.98,hspace=0.15)
    N,k0,alpha,beta=load_parameters(name=str(name))
    fig.suptitle(r'$N={0}, k_0={1}, \alpha={2}, \beta={3}$'.format(N,k0,alpha,beta),fontsize=labelsize)
    ax0  = fig.add_subplot(gs[0,0])
    ax1  = fig.add_subplot(gs[1,0])
    ax0.tick_params(labelsize=ticksize)
    ax1.tick_params(labelsize=ticksize)
    
    f=FortranFile(base_dir+'instances/phi_{0}.dat'.format(name), 'r')
    N,k0,alpha,beta=load_parameters(base_dir='',name=str(name))
    k_list,p_list=TriadList(N,k0)
    
    f_x=FortranFile('realspace/RealSpaceCoordinates_{0}.dat'.format(name), 'r')
    x=np.zeros(2*N,dtype=kr)
    for i in range(2*N):
        x[i]=f_x.read_reals(kr)
    f_x.close()
    del f_x

    RealSpace=FortranFile('realspace/RealSpace_{0}.dat'.format(name), 'r')

    ax0.set_xlim([0.0,2.0*np.pi])
    ax0.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
    ax0.set_xticklabels(["0","$\pi/2$","$\pi$","$3\pi/2$","$2\pi$"])
    ax0.set_ylabel(r"$u(t,x)/u_\text{rms}$", fontsize=labelsize)
    ax0.set_xlabel(r'$x$',fontsize=labelsize)
    
    ax1.set_xticks([0,N//2,N])
    ax1.set_xticklabels([r'$0$',r'$N/2$',r'$N$'],fontsize=labelsize)
    ax1.set_yticks([0,N//2])
    ax1.set_yticklabels([r'$0$',r'$N/2$'],fontsize=labelsize)
    ax1.set_ylabel(r'$p$', fontsize=labelsize)
    ax1.set_xlabel(r'$k$', fontsize=labelsize)
    ax1.set_ylim([0,N//2])
    ax1.set_xlim([0,N])

    im=ax1.imshow(np.zeros((2,2)),interpolation='none',extent=[-2,-1,-2,-1],vmin=0.0,vmax=2.0*np.pi,cmap='rainbow')
    cb=fig.colorbar(im,orientation="vertical",ax=ax1)
    cb.ax.tick_params(labelsize=ticksize)    
    cb.set_label(r'$\varphi_{k,p}$', fontsize=ticksize)
    cb.set_ticks([0.0,0.5*np.pi,np.pi,1.5*np.pi,2.0*np.pi])
    cb.set_ticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])#,fontsize=labelsize)

    ims=[]    
    
    while True:
        im0,status1=plot_RS(ax0,RealSpace,N,x)
        im1,status0=plot_triange(ax1,f,N,k_list,p_list)
        if status0==1 or status1==1:
            break
        else:
            ims.append([im0,im1])

    RealSpace.close
    f.close
    del RealSpace, f
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save('vel_triads_{0}_N_{1}.mp4'.format(name,int(np.log2(N))), dpi=320)
    fig.savefig('figures/vel_triads_{0}_N_{1}.pdf'.format(name,int(np.log2(N))),dpi=150)
    plt.show()
    return 0

if __name__ == '__main__':
    from datetime import datetime
    ta=datetime.now() 
    main(name=1)
    tb=datetime.now()
    print("Done. It took "+str(tb-ta)+".")
    #ta=datetime.now() 
    #main(name=2)
    #tb=datetime.now()
    #print("Done. It took "+str(tb-ta)+".")
