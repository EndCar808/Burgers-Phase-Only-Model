"""Functions and Classes needed for implementation of Ginelli Algorithm.
----------
Contents
----------
- posQR, function that performs QR decomposition with postivie entries on the R diagonal.
- Forward, class for performing the forward integration steps of the Ginelli algorithm.
- block_squish_norm, function for performing CLV convergence steps.
- make_observations, function that uses looker and runner to make many observations.
- make_cupboard, function that makes directories to save observations in.
- max_digit, function for determining max number in list of strings
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import sys
import numpy as np
import xarray as xr
import os
from tqdm.notebook import tqdm

# --------------------------------------------------
# Classes that do most of the work
# --------------------------------------------------

def posQR(M):
    """ Returns QR decomposition of a matrix with positive diagonals on R.
    Parameter, M: Array that is being decomposed
    """
    Q, R = np.linalg.qr(M) # Performing QR decomposition
    signs = np.diag(np.sign(np.diagonal(R))) # Matrix with signs of R diagonal on the diagonal
    Q, R = np.dot(Q, signs), np.dot(signs, R) # Ensuring R Diagonal is positive
    return Q, R

class Forward:
    """Performs forward steps in Ginelli algorithm. Relies on a tangent integrator object"""

    def __init__(self, integrator, tau, oldQ = None):
        """param, integrator: object to integrate both TLE and system itself.
        param, tau: adimensional time between orthonormalisations."""

        self.integrator, self.tau = integrator, tau
        self.step_count = 0

         # Info we need from the integrator
        self.size = integrator.size # size of original + linearised system.

        # Stretched matrix.
        self.P = np.random.rand(self.size, self.size) # Stretched Matrix

        # Initialising orthogonal matrix

        if (oldQ == None):
            eps = 1.e-9
            self.oldQ = eps * np.identity(self.size)

        else:
            self.oldQ = oldQ

        # Stretching rates after QR decomposition
        self.R = np.random.rand(self.size, self.size)

    @property
    def time(self):
        return self.integrator.time
    @property
    def parameter_dict(self):
        ginelli_params = {'tau':self.tau}
        combined_dict = {**self.integrator.parameter_dict , **ginelli_params}
        return combined_dict

    def _step(self):
            """Perform one QR step. Take old Q, stretch it, do a QR decomposition.
            param, location: where we are in phase space.
            param, """

            # Where we are in phase space before ginelli step
            phase_state = self.integrator.state
            time = self.integrator.time

            # Stretching first column
            self.integrator.set_state(phase_state, self.oldQ.T[0]) # First column of Q is ic for TLE
            self.integrator.integrate(self.tau)

            # First column of Stretched matirx
            self.P[:, 0] = self.integrator.tangent_state

            # Stretching the rest of the columns
            for i, column in enumerate(self.oldQ.T[1:]):

                # Reseting to where we were in phase space
                self.integrator.set_state(phase_state, column)
                self.integrator.time = time

                self.integrator.integrate(self.tau)
                self.P[:, i + 1] = self.integrator.tangent_state # i + 1 index needed

            # QR decomposition
            self.oldQ, self.R = posQR(self.P)
            self.step_count += 1

    def run(self, steps, noprog=True):
        """Performs specified number of Ginelli Steps"""
        for i in tqdm(range(steps), disable=noprog):
            self._step()

def block_squish_norm(R_history, A):
    """Function for CLV Convergence Steps.
    Give a timeseries of R's, push A back through the timeseries, normalising as you go."""
    norms = np.linalg.norm(A, axis=0, ord=2)
    normedA = A/norms
    for R in np.flip(R_history.R, axis = 0):
        squishedA = np.linalg.solve(R, normedA)
        norms = np.linalg.norm(squishedA, axis=0, ord=2)
        normedA = squishedA/norms
    return normedA

# --------------------------------------------------
# Generally usefuly
# --------------------------------------------------

def make_observations(runner, lookers, obs_num, obs_freq, noprog=True):
    """Uses looker and runner to make many observations of Ginelli algorithm.
    runner, ginelli Forward object.
    looker, ginelli observer object
    obs_num, how many observations you want.
    obs_freq, ginelli steps between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.run(obs_freq)
        for looker in lookers:
            looker.look(runner)

def make_cupboard():
    """Makes directories to save observations in"""

    for dirName in ['ginelli', 'ginelli/trajectory', 'ginelli/step2', 'ginelli/step2/R', 'ginelli/step2/BLV',
                    'ginelli/step3', 'ginelli/step4', 'ginelli/step5']:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print(f'Made directory {dirName}.\n')
        else: print(f'The directory {dirName} already exists.\n')

def max_digit(strings):
    """Finds the max digit in a list of strings."""
    numbers = []
    for string in strings:
        for s in string:
            if s.isdigit():
                numbers.append(int(s))
    return max(numbers)