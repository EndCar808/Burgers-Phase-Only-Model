""" Lorenz 63  Integrator classes.
Uses adaptive RK54 method.
----------
Contents
----------
= Integrator, class for integrating L63 ODEs.
- TangentIntegrator, class for integrating L63 and corresponding tangent
dynamics simultaneously.
- TrajectoryObserver, class for observing the trajectory of the L63 integration.
- make_observations, function that makes many observations given integrator and observer.
"""
# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import scipy.integrate
import xarray as xr
import sys
from tqdm.notebook import tqdm 

# ------------------------------------------
# Integrator
# ------------------------------------------

class Integrator:

    """Integrates the L63 ODEs using adaptive integration."""
    def __init__(self, a=10, b=8/3, c=28.0,
                 X_init=None, Y_init=None, Z_init=None):

        # Model parameters
        self.a, self.b, self.c = a, b, c
        self.size = 3
        
        self.time = 0 

        # Non-linear Variables
        self.X = np.random.rand() if X_init is None else X_init # Random IC if none given
        self.Y = np.random.rand() if Y_init is None else Y_init
        self.Z = np.random.rand() if Z_init is None else Z_init
        
    def _rhs_X_dt(self, state):
        """Compute the right hand side of nonlinear variables.
        param, state: where we are in phase space"""

        [X, Y, Z] = state

        dXdt = self.a * (Y - X)
        dYdt = (self.c * X) - Y - (X * Z)
        dZdt = (X * Y) - (self.b * Z)
        return np.array([dXdt, dYdt, dZdt])

    def _rhs_dt(self, t, state):
        phase_state = state[:3]
        return [*self._rhs_X_dt(phase_state)]
    
    def integrate(self, how_long):
        """time: how long we integrate for in adimensional time."""
        
        # Where We are
        t = self.time
        IC = self.state
        
        # Integration, uses RK45 with adaptive stepping. THIS IS THE HEART.
        solver_return = scipy.integrate.solve_ivp(self._rhs_dt, (t, t + how_long), IC)
        
        # Updating variables
        new_state = solver_return.y[:,-1]
        self.X = new_state[0]
        self.Y = new_state[1] 
        self.Z = new_state[2] 
        
        self.time = t + how_long

    def set_state(self, x):
        """x is [X, Y, Z]."""
        [self.X, self.Y, self.Z] = x
        
    @property
    def state(self):
        """Where we are in phase space"""
        return np.array([self.X, self.Y, self.Z])

    @property
    def time(self):
        """a-dimensional time"""
        return self.__time
    
    @time.setter   
    def time(self, when):
        self.__time = when

    @property
    def parameter_dict(self):
        param = {
        'a': self.a,
        'b': self.b,
        'c': self.c
        }
        return param

# ------------------------------------------
# TangentIntegrator
# ------------------------------------------

class TangentIntegrator:

    """Integrates the L63 ODEs and it's tangent dynamics simultaneously."""
    def __init__(self, a=10, b=8/3, c=28.0,
                 X_init=None, Y_init=None, Z_init=None, dx_init=None, dy_init=None, dz_init=None):

        # Model parameters
        self.a, self.b, self.c = a, b, c
        self.size = 3
        
        self.time = 0 

        # Non-linear Variables
        self.X = np.random.rand() if X_init is None else X_init # Random IC if none given
        self.Y = np.random.rand() if Y_init is None else Y_init
        self.Z = np.random.rand() if Z_init is None else Z_init

        # TLE Variables
        eps = 1.e-5
        self.dx = eps * np.random.rand() if dx_init is None else dx_init # Random IC if none given
        self.dy = eps * np.random.rand() if dy_init is None else dy_init
        self.dz = eps * np.random.rand() if dz_init is None else dz_init
        
    def _rhs_X_dt(self, state):
        """Compute the right hand side of nonlinear variables.
        param, state: where we are in phase space"""

        [X, Y, Z] = state

        dXdt = self.a * (Y - X)
        dYdt = (self.c * X) - Y - (X * Z)
        dZdt = (X * Y) - (self.b * Z)
        return np.array([dXdt, dYdt, dZdt])

    def _rhs_TX_dt(self, state, tangent_state):
        """Compute the right hand side of the linearised equation."""
        [X, Y, Z] = state
        [dx, dy, dz] = tangent_state

        dxdt = self.a * (dy - dx)
        dydt = (self.c - Z) * dx - dy - (X * dz)
        dzdt = (Y * dx) + (X * dy) - (self.b * dz)
        return np.array([dxdt, dydt, dzdt])

    def _rhs_dt(self, t, state):
        phase_state = state[:3]
        tangent_state = state[3:]
        return [*self._rhs_X_dt(phase_state), *self._rhs_TX_dt(phase_state, tangent_state)]
    
    def integrate(self, how_long):
        """time: how long we integrate for in adimensional time."""
        
        # Where We are
        t = self.time
        IC = np.hstack((self.state, self.tangent_state))
        
        # Integration, uses RK45 with adaptive stepping. THIS IS THE HEART.
        solver_return = scipy.integrate.solve_ivp(self._rhs_dt, (t, t + how_long), IC)
        
        # Updating variables
        new_state = solver_return.y[:,-1]
        self.X = new_state[0]
        self.Y = new_state[1] 
        self.Z = new_state[2] 
        self.dx = new_state[3]
        self.dy = new_state[4]
        self.dz = new_state[5]
        
        self.time = t + how_long

    def set_state(self, x, tangent_x):
        """x is [X, Y, Z]. tangent_x is [dx, dy, dz]"""
        [self.X, self.Y, self.Z] = x
        [self.dx, self.dy, self.dz] = tangent_x
        
    @property
    def state(self):
        """Where we are in phase space"""
        return np.array([self.X, self.Y, self.Z])

    @property
    def tangent_state(self):
        """Where we are in tangent space"""
        return np.array([self.dx, self.dy, self.dz])

    @property
    def time(self):
        """a-dimensional time"""
        return self.__time
    
    @time.setter   
    def time(self, when):
        self.__time = when

    @property
    def parameter_dict(self):
        param = {
        'a': self.a,
        'b': self.b,
        'c': self.c
        }
        return param
    
# ------------------------------------------
# TrajectoryObserver
# ------------------------------------------
    
class TrajectoryObserver():
    """Observes the trajectory of L63 ODE integrator. Dumps to netcdf."""

    def __init__(self, integrator, name='L63 Trajectory'):
        """param, integrator: integrator being observed."""

        # Need knowledge of the integrator
        self._parameters = integrator.parameter_dict

        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.state_obs = []
        self.dump_count = 0
        self.integrator = integrator

    def look(self, integrator):
        """Observes trajectory of L96 trajectory"""

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
        
# ------------------------------------------
# make_observations
# ------------------------------------------

def make_observations(runner, looker, obs_num, obs_freq, noprog=False):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.integrate(obs_freq)
        looker.look(runner)