#Agustin Arguedas

from scipy.io import FortranFile
kr='float64'

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
labelsize=25
ticksize=22

import matplotlib.animation as animation
from plot_miguel import u_rms

animation.writers['ffmpeg'](
    fps=30,
    metadata={'title': ''},
    codec='libx264',
    extra_args=[  # '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        '-crf', '19',
        '-profile:v', 'high',
        '-threads', '8'])

def vel_animation(i,j):
    import matplotlib.animation as animation    
    name_i='_'+str(i)
    name_j='_'+str(j)
    par=np.loadtxt('parameters/Parameters' +name_i+'.dat').flatten()
    N_i=int(par[0])
    k0_i=int(par[1])
    alpha_i=float(par[2])
    beta_i=float(par[3])
    par=np.loadtxt('parameters/Parameters' +name_j+'.dat').flatten()
    N_j=int(par[0])
    k0_j=int(par[1])
    alpha_j=float(par[2])
    beta_j=float(par[3])
    RealSpace_i=FortranFile('./realspace/RealSpace{0}.dat'.format(name_i), 'r')
    RealSpace_j=FortranFile('./realspace/RealSpace{0}.dat'.format(name_j), 'r')
    f_x_i=FortranFile('./realspace/RealSpaceCoordinates{0}.dat'.format(name_i), 'r')
    f_x_j=FortranFile('./realspace/RealSpaceCoordinates{0}.dat'.format(name_j), 'r')
    x_i=np.empty(2*N_i,dtype='float64')
    for k in range(2*N_i):
        x_i[k]=f_x_i.read_reals(kr)
    f_x_i.close()
    x_j=np.empty(2*N_j,dtype='float64')
    for k in range(2*N_j):
        x_j[k]=f_x_j.read_reals(kr)
    f_x_j.close()
    
    fig = plt.figure(figsize=(9.,6.),facecolor = 'white')
    fig.subplots_adjust(top=0.93,bottom=0.12,left=0.13,right=0.97)
    fig.suptitle(r'$N={0}, \alpha={1}, \beta={2}$'.format(N_i,alpha_i,beta_i),fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([0.0,2.0*np.pi])
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
    ax.set_xticklabels(["0","$\pi/2$","$\pi$","$3\pi/2$","$2\pi$"])
    ax.set_ylabel(r"$u(t,x)/u_\text{rms}$", fontsize=labelsize)
    ax.set_xlabel(r'$x$',fontsize=labelsize)
    ax.plot([-1,-2],[0,0],'b',label=r'$k_0={0}$'.format(k0_i))
    ax.plot([-1,-2],[0,0],'r',label=r'$k_0={0}$'.format(k0_j))
    ax.legend(fontsize=labelsize,loc=1,ncol=2,fancybox=True, framealpha=0.5)
    
    ims=[]    
    
    while True:
        try:
            RS_i=RealSpace_i.read_reals(kr).reshape((2*N_i), order="F")
            RS_j=RealSpace_j.read_reals(kr).reshape((2*N_j), order="F")
            im0,=ax.plot(x_i,RS_i,'b')
            im1,=ax.plot(x_j,RS_j,'r')
            ims.append([im0,im1])
        except:
            break
        
    ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True)
    #plt.show()
    RealSpace_i.close
    RealSpace_j.close

    ani.save('Compare_N'+str(int(np.log2(N_i)))+'.mp4', dpi=320)
    
if __name__=='__main__':
   vel_animation(0,1)