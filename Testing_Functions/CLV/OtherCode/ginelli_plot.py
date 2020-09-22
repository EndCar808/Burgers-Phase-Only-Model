#---------------------------------------------------------------
# Contains various functions for processing and plotting 
# the results of a Ginelli algorithm run.
#---------------------------------------------------------------
#---------------------------------------------------------------
# Contents:
# - min_pos, function for finding minimum positive LE index.
# 
# - LE_average, function for computing FTLE average
# over a longer Ltau.
#
# - spectra_parameter, function that plost LE spectra for a 
# varying parameter
# - density plot, NEEDS FINISHIN
#---------------------------------------------------------------

import matplotlib.pyplot as plt

#---------------------------------------------------------------
# Processing Utilities
#---------------------------------------------------------------

#---------------------------------------------------------------
# Minimum positive LE
#---------------------------------------------------------------
def spectra(data, geometry='C'):
    """Returns mean Lyapunov exponents.
    """
    if (geometry == 'C'):
        spectra = data.ftcle.mean(dim='time', skipna=True)
    else: # then FTBLE
        spectra = data.ftble.mean(dim='time', skipna=True)
    return spectra
#---------------------------------------------------------------
# Minimum positive LE
#---------------------------------------------------------------
def min_pos(LE):
    """Finds the minimum non-negative LE index. (So neutral manifold bunched with
    positive).
    LE, xarray of Lyapunov spectra.
    """
    return LE.where(LE>=0).dropna(dim='le_index').le_index.max().item()

#---------------------------------------------------------------
# Longer FTLE average
#---------------------------------------------------------------

def LE_average(data, L, save='None'):
    """Returns xarray with FTLEs being averaged over long times. DESIGNED FOR PROCESSED DATA"""
    ftble = data.ftble.rolling(time = L).mean()[L - 1::L]
    ftcle = data.ftcle.rolling(time = L).mean()[L - 1::L]
    avg_data = data.sel(time = ftble.time) # Picking out data and average time steps.
    avg_data.ftble.values = ftble.values # Updating LEs
    avg_data.ftcle.values = ftcle.values # Updating LEs

    # Adding Attributes
    avg_data.attrs.update({'L' : L})
    Ltau = L * avg_data.attrs['tau']
    avg_data.attrs.update({'Avg Time' : Ltau})
    
    if (save != None):
        avg_data.to_netcdf(save)
    
    return avg_data

#---------------------------------------------------------------
# Plotting Utilities
#---------------------------------------------------------------

#---------------------------------------------------------------
# Spectra Comparision Plot
#---------------------------------------------------------------

def spectra_parameter(data_list, parameter_key, geometry='C', save = 'None'):
    """Plots Lyapunov spectra for a varying parameter.
    data_list, list of data arrays containing lyapunov observations
    parameter_key, the parameter from data.attrs that we're investigating.
    geometry, CLEs or BLEs.
    save, file save name
    """

    for data in data_list:
        
        attr = data.attrs[parameter_key]
       
        if (geometry == 'C'):
            spectra = data.ftcle.mean(dim='time', skipna=True)
            label = f'${attr:.2f}$'
        else: # then FTBLE
            spectra = data.ftble.mean(dim='time', skipna=True)
            label = f'${attr:.2f}$'

        plt.plot(spectra, label=label)

    plt.legend(title = f'{parameter_key}')
    plt.title(f'FT{geometry}LE Means')
    plt.xlabel('LE Index') 
    plt.grid()
    
    if (save == 'None'):
        plt.show()
    else:
        print('Saving')
        plt.savefig(save, dpi=1200)
        plt.show()
        
#---------------------------------------------------------------
# Density Plot NEEDS FINISHING
#---------------------------------------------------------------

def density(data_list, le, ftle='C', save = 'None'):
    """Plots Density using KDE, designed to compare for different h"""
    for data in data_list:
        Ltau = data.attrs['tau'] * 1 # for now L = 1
        h = data.attrs[variable]
        
        # FTCLE or FTBLE?
        if (ftle == 'C'):
            FTLE = data.FTCLE.dropna(dim= 'time', how= 'all').sel(le_index = le).values
            label = f'$h={h:.2f}$'
        else: # then FTBLE
            FTLE = data.FTCLE.dropna(dim= 'time', how= 'all').sel(le_index = le).values
            label = f'$h={h:.2f}$'
            
        x_d = np.linspace(FTLE.min() - 1, FTLE.max() + 1, 100) # Grid we evaluate PDF on
        kde = gaussian_kde(FTLE) # KDE. Using Gaussian kernels with default bandwidth, don't know how it decided bw?
        pdf = kde.evaluate(x_d)
        plt.plot(x_d, pdf, label = label) 
        
    plt.legend()
    plt.xlabel(f'FT{ftle}LE')
    plt.ylabel('$\\rho$')
    plt.title(f'FT{ftle}LE {le} Density Comparison, $L\\tau =' + f"{Ltau:.2f}" + '$')
    
    if (save == 'None'):
        plt.show()

    else:
        print('Saving')
        plt.savefig(f'Effect-of-Coupling-Strength/FTCLE-Densities/FT{ftle}LE_{le}.png', dpi=1200)
        plt.show()
        

