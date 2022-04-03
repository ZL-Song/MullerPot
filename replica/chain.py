"""Implements the chain-of-replicas.

Authors: Zilin Song.
"""

import abc, numpy
class ReplicaChainBase(abc.ABC):
    """The abstract class for projecting Cartersian space to Replica space."""

    @property
    @abc.abstractmethod
    def cartersian_coors(self):
        """Read Only: Return the cartersian coordinates of the replica chain."""
    
    @property
    @abc.abstractmethod
    def replicas_vec(self):
        """Return the replcia chain vector (nrep, ndof)."""
    
    @replicas_vec.setter
    @abc.abstractmethod
    def replicas_vec(self, replicas:numpy.ndarray):
        """Set the replica chain vector (nrep, ndof)."""

    @classmethod
    @abc.abstractmethod
    def rms(self):
        """Get the RMS values of adjacent replicas.
        possibily best-fit and mass-weighted.
        
        Return
        ------
        rms: numpy.ndarray
           The RMS values between adjacent replicas.
        """

class Replica2DChain(ReplicaChainBase):
    """The 2D chain-of-replicas.
    
    Parameters
    ----------
    rms_best_fit: bool, default=False
        If the best-fit RMS should be computed.
        Ignored on 2D chain-of-replicas.

    rms_mass_weighted:bool, default=False
        If the mass-weighted RMS should be computed.
        Ignored on 2D chain-of-replicas.
    """
    
    def __init__(self, replicas:numpy.ndarray, rms_best_fit:bool = False, rms_mass_weighted:bool = False):
        self._replicas_vec = replicas
        self._rms_best_fit = rms_best_fit
        self._rms_mass_wgh = rms_mass_weighted
    
    @property
    def cartersian_coors(self):
        """Read Only: Return the cartersian coordinates of the replica chain."""
        return self._replicas_vec
    
    @property
    def replicas_vec(self):
        """Return the replcia chain vector (nrep, ndof)."""
        return self._replicas_vec
    
    @replicas_vec.setter
    def replicas_vec(self, replicas:numpy.ndarray):
        """Set the replica chain vector (nrep, 2)."""

        if replicas.shape[1] != 2:
            raise ValueError('Replica2DChain() allows only 2 degrees-of-freedom in each replica.')

        else:
            self._replicas_vec = replicas
    
    def rms(self):
        """Get the RMS values of adjacent replicas.
        possibily best-fit and mass-weighted.
        
        Return
        ------
        rms: numpy.ndarray
           The RMS values between adjacent replicas.
        """
        rep0s = self._replicas_vec[0:-1, :]  #  first reps
        rep1s = self._replicas_vec[1:  , :]  # second reps

        rms = numpy.sqrt(numpy.sum((rep1s-rep0s)**2, axis=1))  # RMS between neighboring 2d points.

        return rms