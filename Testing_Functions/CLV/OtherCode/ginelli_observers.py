"""Observer classes for the Ginelli Algorithm.
----------
Contents
----------
- RMatrixObserver, class for observing the R matrices. Note the FTBLEs can be calculated from the diagonals of these Rs.
- BLVMatrixObserver, class for observing the R matrices.
- LyapunovObserver, class for observing FTBLEs, FTCLEs, CLVs, BLVS. Used within Ginelli code, not specified at top of script like other observers.
- L63TrajectoryObserver, class for observing trajectory of L63 System.
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

# ----------------------------------------
# RMatrixObserver
# ----------------------------------------

class RMatrixObserver:
    """Observes the R Matrix in the forward Ginelli Steps."""

    def __init__(self, ginelli):
        """param, ginelli: Ginelli Forward Stepper being obseved."""

        # Knowledge associated with this observer
        self.name = 'R' # will be used in file save
        self.ginelli = ginelli # stepper we're associated with
        self.parameters = ginelli.parameter_dict
        self.dump_count = 0

        # R Observation log
        self.R_obs = []
        self.time_obs = []

    def look(self, ginelli):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(ginelli.time)

        # Making Observations
        self.R_obs.append(ginelli.R.copy())

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.R_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['R'] = xr.DataArray(self.R_obs, dims=['time', 'row', 'column'], name='R',
                                coords = {'time': _time})

        return xr.Dataset(dic, attrs=self.parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.R_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.R_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1

# ----------------------------------------
# BLVMatrixObserver
# ----------------------------------------

class BLVMatrixObserver:
    """Observes the Q Matrix in the forward Ginelli Steps."""

    def __init__(self, ginelli):
        """param, ginelli: Ginelli Forward Stepper being obseved."""

        # Knowledge from Ginelli object
        self.ginelli = ginelli # stepper we're associated with
        self.parameters = ginelli.parameter_dict
        self.le_index = np.arange(1, 1 + ginelli.size)

        # Info
        self.name = 'BLV' # will be used in file save
        self.dump_count = 0

        # BLV Observation log
        self.BLV_obs = []
        self.time_obs = []

    def look(self, ginelli):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(ginelli.time)

        # Making Observations
        self.BLV_obs.append(ginelli.oldQ.copy())

    @property
    def observations(self):
        if (len(self.BLV_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['BLV'] = xr.DataArray(self.BLV_obs, dims=['time', 'row', 'le_index'], name='BLV',
                                coords = {'time': _time, 'le_index': self.le_index})

        return xr.Dataset(dic, attrs=self.parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.BLV_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.BLV_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1

# ----------------------------------------
# LyapunovObserver
# ----------------------------------------

class LyapunovObserver:
    """Observes CLVs, BLVs, FTCLEs and FTBLEs. Designed to be used in final step of Ginelli algorithm."""

    def __init__(self, metadata, number_files):
        """param, metadata: Information we want associated with this data file.
        param, numer_files: Number of files observations are dumped into"""

        # Knowledge
        self.parameters = metadata
        self.name = 'LyapObs'
        self.dump_count = number_files

        # Observation log
        self.time_obs = []
        self.CLV_obs = []
        self.BLV_obs = []
        self.ftcle_obs = []
        self.ftble_obs = []

    def look(self, time, CLV, BLV, ftcle, ftble):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(time)

        # Making Observations
        self.CLV_obs.append(CLV)
        self.BLV_obs.append(BLV)
        self.ftcle_obs.append(ftcle)
        self.ftble_obs.append(ftble)

    @property
    def observations(self):
        if (len(self.CLV_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        _le_index = np.arange(1, self.BLV_obs[0].shape[0] + 1)
        dic['ftble'] = xr.DataArray(self.ftble_obs, dims=['time', 'le_index'], name='FTBLE',
                                   coords={'time': _time, 'le_index': _le_index})
        dic['ftcle'] = xr.DataArray(self.ftcle_obs, dims=['time', 'le_index'], name='FTCLE',
                                   coords={'time': _time, 'le_index': _le_index})
        dic['CLV'] = xr.DataArray(self.CLV_obs, dims=['time', 'row', 'le_index'], name='CLV',
                                coords = {'time': _time, 'le_index': _le_index})
        dic['BLV'] = xr.DataArray(self.BLV_obs, dims=['time', 'row', 'le_index'], name='BLV',
                                coords = {'time': _time, 'le_index': _le_index})

        ds = xr.Dataset(dic, attrs=self.parameters)
        return ds.reindex(time = ds.time[::-1]) # Return reversed array as we observe backwards

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.CLV_obs = []
        self.BLV_obs = []
        self.ftcle_obs = []
        self.ftble_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.BLV_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count -=1 # count down as we observe backwards


# ------------------------------------------
# L63TrajectoryObserver
# ------------------------------------------
    
class L63TrajectoryObserver():
    """Observes the trajectory of L63 ODE integrator. Dumps to netcdf."""

    def __init__(self, ginelli):
        """param, integrator: integrator being observed."""
        
        self.name = 'Trajectory'
        
        # Knowledge from Ginelli object
        self.integrator = ginelli.integrator # stepper we're associated with

        # Need knowledge of the integrator
        self._parameters = ginelli.integrator.parameter_dict

        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.state_obs = []
        self.dump_count = 0

    def look(self, ginelli):
        """Observes trajectory of L63"""
        
        integrator = ginelli.integrator

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.state_obs.append(integrator.state.copy())

    def wipe(self):
        self.time_obs = [] # Times we've made observations
        self.state_obs = []

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.state_obs) == 0):
            print('I have no observations! :(')
            return

        _time = self.time_obs
        x = xr.DataArray(np.array(self.state_obs)[:, 0], dims=['Time'], name = 'X', coords = {'Time': _time})
        y = xr.DataArray(np.array(self.state_obs)[:, 1], dims=['Time'], name = 'Y', coords = {'Time': _time})
        z = xr.DataArray(np.array(self.state_obs)[:, 2], dims=['Time'], name = 'Z', coords = {'Time': _time})
        trajectory = xr.Dataset({'X': x, 'Y': y, 'Z': z})

        return trajectory
    
    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.time_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1