#Agustin Arguedas

from scipy.io import FortranFile
kr='float64'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
labelsize=25
ticksize=22
from plot_routines import load_parameters, load_RS

import matplotlib.animation as animation

animation.writers['ffmpeg'](
    fps=30,
    metadata={'title': ''},
    codec='libx264',
    extra_args=[  # '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        '-crf', '19',
        '-profile:v', 'high'])

def vel_animation(subname='',i=1):
    import matplotlib.animation as animation    
    name=subname+'_'+str(i)
    N,k0,alpha,beta,n_max,n_bins, x_lim, dur_norm=load_parameters(name)
    RealSpace=FortranFile('../realspace/RealSpace_{0}.dat'.format(name), 'r')
    #RealSpace=np.loadtxt('realspace/RealSpace'+name+'.dat')
    Grad=FortranFile('../realspace/Gradient_{0}.dat'.format(name), 'r')
    #Grad=np.loadtxt('realspace/Gradient'+name+'.dat')
    x=load_RS(name,'RealSpaceCoordinates')
    fig = plt.figure(figsize=(9.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.12,left=0.13,right=0.97)
    fig.suptitle(r'$N={0}, k_0={1}, \alpha={2}, \beta={3}$'.format(N,k0,alpha,beta),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([0.0,2.0*np.pi])
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
    ax.set_xticklabels(["0","$\pi/2$","$\pi$","$3\pi/2$","$2\pi$"])
    ax.set_ylabel(r"$u(t,x)/u_\text{rms}$", fontsize=labelsize)
    ax.set_xlabel(r'$x$',fontsize=labelsize)
    
    #steps=RealSpace[:,0].size
    
    ims=[]    
    
    while True:
        try:
            RS=RealSpace.read_reals(kr).reshape((2*N), order="F")
            G=Grad.read_reals(kr).reshape((2*N), order="F")
            im0,=ax.plot(x,RS,'b')
            #im1,=ax.plot(x,G,'r')
            ims.append([im0])
        except:
            break
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    plt.show()
    RealSpace.close
    Grad.close
    ani.save('Velocity_field'+name+'_N'+str(int(np.log2(N)))+'.mp4', dpi=320)
    
if __name__=='__main__':
    vel_animation(subname='N15_k0',i=11)