#utilities needed for plotting routines
import numpy as np
from scipy.io import FortranFile
kr='float64'
ki='int64'
from scipy import stats

#Loading routines
def load_Hists(name,
               subname,
               Norm=False,
               reduction=False,
               red_factor=10,
               base_dir='../'
               ):
    pars=load_parameters(name=name,base_dir=base_dir)

    f=FortranFile(base_dir+'histograms/bins_'+subname+'_'+name+'.dat', 'r')
    bins=f.read_reals(kr)
    f.close

    f=FortranFile(base_dir+'histograms/Hist_'+subname+'_'+name+'.dat', 'r')
    H=f.read_reals(ki)
    f.close

    bins= (bins[1:] +bins[:-1] )*0.5

    if reduction == True :
        H=np.sum(np.resize(H,(pars['n_bins']//red_factor,red_factor)),axis=1)
        bins=np.average(np.resize(bins,(pars['n_bins']//red_factor,red_factor)),axis=1)

    dx=(bins[1]-bins[0])
    args =np.argwhere(H != 0)
    bins=bins[args]
    H=H[args]
    H=H/np.sum(H*dx)
    if Norm==True:
        var=np.sqrt(np.sum(H*bins**2*dx))
        H*=var
        dx/=var
        bins/=var
    else:
        H/=float(pars['dur_norm'])
    return H, bins, dx

def load_parameters(base_dir='../',
                    name='1'
                    ):
    par=np.loadtxt(base_dir+'parameters/Parameters_{0}.dat'.format(name))
    N=int(par[0])
    k0=int(par[1])
    alpha=float(par[2])
    beta=float(par[3])
    n_max=int(par[4])
    n_bins=int(par[5])
    x_lim=float(par[6])
    dur_norm=float(par[7])
    pars={
    'N':N,
    'k0':k0,
    'alpha':alpha,
    'beta':beta,
    'n_max':n_max,
    'n_bins':n_bins,
    'x_lim':x_lim,
    'dur_norm':dur_norm
    }    
    return pars

def load_RS(name,subname,base_dir='../'):
    f=FortranFile(base_dir+'realspace/'+subname+'_'+name+'.dat', 'r')
    RS=f.read_reals(kr)
    f.close
    return RS

def load_stats(name,subname,base_dir='../'):
    f=FortranFile(base_dir+'statistics/'+subname+'_'+name+'.dat', 'r')
    stats_r=f.read_reals(kr)
    f.close
    return stats_r

def load_statistics(l_name,
                    to_load,
                    r_index,
                    base_dir='../'
                    ):
    length=len(to_load)
    flatness=np.zeros(length,dtype='float64')
    skewness=np.zeros(length,dtype='float64')
    alpha_list=np.zeros(length,dtype='float64')
    for j in range(length):
        i=to_load[j]
        name=l_name+'_'+str(i)
        pars=load_parameters(name=name,base_dir=base_dir)
        subname='increment_statistics'
        IS=load_stats(name,subname,base_dir=base_dir)
        IS=IS.reshape(pars['N'],pars['n_max']).T
        moment2=IS[1,r_index]
        moment3=IS[2,r_index]
        moment4=IS[3,r_index]
        alpha_list[j]=pars['alpha']
        flatness[j]=moment4/moment2**2
        skewness[j]=moment3/np.sqrt(moment2)**3
    return alpha_list,skewness,flatness

#Carry out computations
def central_derivative(f,x):
    result=np.empty(f.size)
    result[1:-1]=(f[2:]-f[:-2])/(x[2:]-x[:-2])
    result[0]=(f[1]-f[0])/(x[1]-x[0])
    result[-1]=(f[-1]-f[-2])/(x[-1]-x[-2])
    return result

def get_log_derivative(l_name,
                       i=None,
                       moment=4,
                       normed=False,
                       base_dir='../'
                       ):
    if i==None:
        name=l_name
    else:
        name=l_name+'_'+str(i)
    pars=load_parameters(name=name,base_dir=base_dir)
    urms=u_rms(N=pars['N'],k0=pars['k0'],alpha=pars['alpha'],beta=pars['beta'])
    x=load_RS(name,'RealSpaceCoordinates')
    x=x[1:pars['N']+1]
    x_C=np.copy(x)
    x=np.log10(x)
    subname='increment_statistics'
    IS=load_stats(name,subname,base_dir=base_dir)
    IS=IS.reshape(pars['N'],pars['n_max']).T
    if normed==False:
        exp_val=np.log10(np.abs(IS[moment-1]))
    elif normed==True:
        exp_val=np.log10(np.abs(IS[moment-1])/urms**moment)
    else:
        print('False normed option.')
        return 1
    return x_C, central_derivative(exp_val,x)

def get_scaling_exponent(l_name,
                         i=None,
                         moment=4,
                         normed=False,
                         scale=5,
                         base_dir='../'
                         ):
    if i==None:
        name=l_name
    else:
        name=l_name+'_'+str(i)
    pars=load_parameters(name=name,base_dir=base_dir)
    urms=u_rms(N=pars['N'],k0=pars['k0'],alpha=pars['alpha'],beta=pars['beta'])
    x=load_RS(name,'RealSpaceCoordinates')
    int_len=2.0*np.pi/pars['k0']
    print('Using lenght '+str(x[scale]/int_len))
    x=np.log10(x[1:pars['N']+1])
    subname='increment_statistics'
    IS=load_stats(name,subname)
    IS=IS.reshape(pars['N'],pars['n_max']).T
    if normed==False:
        exp_val=np.log10(np.abs(IS[moment-1]))
    elif normed==True:
        exp_val=np.log10(np.abs(IS[moment-1])/urms**moment)
    else:
        print('False normed option.')
        return 1
    scaling_exponent=(exp_val[scale]-exp_val[scale-1])/(x[scale]-x[scale-1])
    return scaling_exponent

def get_scaling_exponent_list(l_name,
                              i,
                              moment=np.array([4]).astype(np.int),
                              normed=False,
                              scale=5,
                              base_dir='../'
                              ):
    nums=moment.size
    exp_list=np.empty(nums,dtype='float64')
    for j in range(nums):
        exp_list[j]=get_scaling_exponent(l_name,i,moment=moment[j],normed=normed,scale=scale,base_dir=base_dir)
    return exp_list

def get_slope(name,
              i,
              moment=4,
              normed=False,
              scale=10,
              base_dir='../'
              ):
    pars=load_parameters(name=name+'_{0}'.format(i),base_dir=base_dir)
    urms=u_rms(N=pars['N'],k0=pars['k0'],alpha=pars['alpha'],beta=pars['beta'])
    x=load_RS(name+'_{0}'.format(i),'RealSpaceCoordinates')
    x=x[1:pars['N']+1]
    x=np.log10(x)
    subname='increment_statistics'
    IS=load_stats(name+'_{0}'.format(i),subname)
    IS=IS.reshape(pars['N'],pars['n_max']).T
    if normed==False:
        exp_val=np.log10(np.abs(IS[moment-1]))
    elif normed==True:
        exp_val=np.log10(np.abs(IS[moment-1])/urms**moment)
    else:
        print('False normed option.')
        return 1
#    print x
#    x=x[scale]
#    exp_val=exp_val[scale]
#    print x
    #slope, intercept, r_value, p_value, std_err = stats.linregress(x,exp_val)
    return slope,intercept

def get_slope_list(l_name,
                   i,
                   moment=np.array([4]).astype(np.int),
                   normed=False,
                   scale=5,
                   base_dir='../'
                   ):
    nums=moment.size
    exp_list=np.empty(nums,dtype='float64')
    for j in range(nums):
        x,slope=get_log_derivative(l_name,#Changed this here!
                       i=i,
                       moment=moment[j],
                       normed=normed,
                       base_dir='../'
                       )#get_slope(l_name,i,moment=moment[j],normed=normed,scale=scale,base_dir=base_dir)
        slope=slope[16]
        exp_list[j]=slope
    return exp_list

def theoretical_energy(N=1024,
                       k0=4,
                       alpha=1.0,
                       beta=1.0):
    k=np.arange(k0+1,N+1,dtype='float64')
    ak=k**(-alpha)*np.exp(-beta*(2.0*k/float(N))**2)
    return 2.0/float(N)*np.sum(ak**2)
    
def u_rms(N=1024,
          k0=4,
          alpha=1.0,
          beta=1.0):
    return np.sqrt(theoretical_energy(N=N,k0=k0,alpha=alpha,beta=beta))