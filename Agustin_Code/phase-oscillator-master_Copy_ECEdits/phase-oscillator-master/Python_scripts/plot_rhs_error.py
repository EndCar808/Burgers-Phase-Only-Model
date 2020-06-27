#Agustin Arguedas

def main():
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    import numpy as np
    import matplotlib.pyplot as plt
    
    labelsize=25
    ticksize=22
    Load=np.loadtxt('rhs_test.dat')
    N=Load[:,0]
    time=Load[:,1]
    time_fft=Load[:,2]
    error=Load[:,3]
    
    time=np.where(time==0.0,np.finfo(np.float64).eps,time)
    time_fft=np.where(time_fft==0.0,np.finfo(np.float64).eps,time_fft)
        
    fig = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig.subplots_adjust(top=0.89,bottom=0.18,left=0.13,right=0.95)
    fig.suptitle("Time",fontsize=labelsize)
    ax  = fig.add_subplot(111)
    ax.set_xlabel(r'$\log_2 N$',fontsize=labelsize)
    ax.set_yscale('log')
    ax.tick_params(labelsize=ticksize)
    ax.plot(np.log2(N),time,label=r'DO-Loop')
    ax.plot(np.log2(N),time_fft,label=r'FFT')
    ax.legend(fontsize=labelsize, loc=2)
    fig.savefig('rhs_Time.pdf',dpi=150)
    
    fig2 = plt.figure(figsize=(6.,4.),facecolor = 'white')
    fig2.subplots_adjust(top=0.75,bottom=0.17,left=0.13,right=0.96)
    fig2.suptitle("Error"+'\n'+r'$\vert (u\ast u)-(u\ast u)_{FFT} \vert/\vert (u\ast u)\vert$',fontsize=labelsize)
    ax2  = fig2.add_subplot(111)
    ax2.set_xlabel(r'$\log_2 N$',fontsize=labelsize)
    ax2.tick_params(labelsize=ticksize)
    ax2.plot(np.log2(N),error)
    ax2.set_yscale('log')
    fig2.savefig('rhs_Error.pdf',dpi=150)
    #plt.show()
    
if __name__=='__main__':
    main()