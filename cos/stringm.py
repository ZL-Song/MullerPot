"""Implements the string method and the string intepolators.

Authors: Zilin Song.
"""

import numpy, abc, copy
from cos.optimizers import OptimizerBase, SteepestDecent, AdaptiveMomentum
from replica.chain import ReplicaChainBase
from scipy.interpolate import CubicSpline 

class StringMethod(object):
    """Class for managing the chain of replicas using the string method.

    Parameters
    ----------
    chain: ReplicaChainBase
        The chain of string images.

    intpol_method: str or a class instance that inherits StrmInterpolatorBase, default: 'cspline'
        The method to interpolate the string images. Allowed values are
            "cspline": Interpolate using StrmCubicSplineInterpolator().
        
        > Note, if a class instance that inherits StrmInterpolatorBase() is passed,  
        > The interpolators will be deep-copied from this instance.
    
    optimizer: str or a class instance that inherits OptimizerBase, default: 'sd'
        The optimization method for string evolution. Allowed values are
            "sd":   steepest decent optimizer;
            "adam": adaptive momentum optimizer.

    Attributes
    ----------
    chain: ReplicaChainBase
        The chain of string images.

    nrep: int
        The number of replicas in string. 

    ndof: int
        The number of degrees of freedom in each string replica.

    """
    def __init__(self, chain:ReplicaChainBase, intpol_method:str='cspline', optimizer:str='sd'):
        self.chain = chain                              # Chain-of-States
        self.nrep  = self.chain.replicas_vec.shape[0]   # No. replicas.
        self.ndof  = self.chain.replicas_vec.shape[1]   # No. degrees of freedom.

        # Determine Interpolator.
        if isinstance(intpol_method, StrmInterpolatorBase):
            self._intpol_method = intpol_method

        elif isinstance(intpol_method, str):

            if intpol_method.lower() == 'cspline':
                self._intpol_method = StrmCubicSplineInterpolator()

            else:
                raise NotImplementedError(f"The interpolation method: {intpol_method} is not implemented.")

        else:
            raise ValueError(f"The interpolator instance must inherit StrmInterpolatorBase.")

        # Determine Optimizer
        if isinstance(optimizer, OptimizerBase):
            self._optimizer = optimizer

        elif isinstance(optimizer, str):

            if optimizer.lower() == 'sd':
                self._optimizer = SteepestDecent()

            elif optimizer.lower() == 'adam':
                self._optimizer = AdaptiveMomentum()

            else:
                raise NotImplementedError(f"The optimization method: \'{optimizer}\' is not implemented.")

        else: 
            raise ValueError(f"The optimizer instance must inherit OptimizerBase.")

        self._init_reparam  = False  # If reparametrization has been done at least once.
        self._interpolators = None   # The list of interpolator for each degrees of freedom.

    def _fit_intpol(self, x, y):
        """Make and return a fitted interpolator."""
        intpol = copy.deepcopy(self._intpol_method)
        intpol.fit(x, y)
        
        return intpol

    def _get_alpha_vec(self):
        """Get the normalized cartersian arclengthes (the alphas) of the string images.
        interrep_arclen is computed as the RMS between adjacent replicas.
        """
        interrep_arclen = self.chain.rms()
        interrep_arclen = numpy.insert(interrep_arclen, 0, 0.)                  # initial replica's rms = 0.
        alpha_vec = numpy.cumsum(interrep_arclen) / numpy.sum(interrep_arclen)  # normalize to [0, 1].
        
        return  alpha_vec

    def _reparametrize(self):
        """Interpolate the string images and re-distribute for equal arclength."""
        
        alpha_vec     = self._get_alpha_vec()
        alpha_vec_new = numpy.linspace(0., 1., num=self.nrep)  # redistributed for equal arclength.
        
        self._init_reparam = True

        self._interpolators = []
        replicas_vec_new = numpy.zeros(self.chain.replicas_vec.shape)
        
        # reparametrize each dimension of string images.
        for idof in range(self.ndof):
            replicas_idof = self.chain.replicas_vec[:, idof]
            intpol = self._fit_intpol(alpha_vec, replicas_idof)
            replicas_vec_new[:, idof] = intpol.transform(alpha_vec_new)
            self._interpolators.append(intpol)

        self.chain.replicas_vec = replicas_vec_new
    
    def _get_string_tangent(self):
        """Get the gradient matrix of the parametrized string. 
        Should only be called after the initial self._reparametrized().
        """
        tangent_vec = numpy.zeros(self.chain.replicas_vec.shape)
        alpha_vec   = self._get_alpha_vec()

        # get the tangent vectors for each dimension.
        for idof in range(self.ndof):
            intpol = self._interpolators[idof]
            tangent_idof = intpol.get_gradient(alpha_vec)
            tangent_vec[:, idof] = tangent_idof

        return tangent_vec

    def _get_projected_force(self, force_vec:numpy.ndarray, tangent_vec:numpy.ndarray):
        """Project the forces perpendicular to the string tangent vectors."""
        # sanity check for force vectors
        if force_vec.shape != self.chain.replicas_vec.shape:
            raise ValueError(f"Inconsistent force_vec passed.\nforce_vec.shape {force_vec.shape}, \
                               chain.replicas_vec.shape {self.chain.replicas_vec.shape}.")
        
        # project force at each replica.
        proj_force_vec = numpy.zeros(force_vec.shape)

        for irep in range(self.nrep): # loop over all replicas.

            if irep == 0 or irep == self.nrep-1:   # do not project the ending replicas.
                proj_force_vec[irep, :] = force_vec[irep, :]

            else:                                  # but project all replicas in between.
                tangent_irep = tangent_vec[irep, :]
                force_irep   = force_vec[irep, :]

                # string tangent vectors F_tan = (F @ v_tan) / (||v_tan||^2) * v_tan
                # the projection perpendicular to F_tan is F_norm = F - F_tan.
                proj_force_irep = force_irep - \
                    numpy.sum(force_irep * tangent_irep) / numpy.sum(tangent_irep**2) * tangent_irep

                proj_force_vec[irep, :] = proj_force_irep

        return proj_force_vec

    def evolve(self, force_vec:numpy.ndarray, reparametrize:bool=True):
        """Envolve the string using one optimizer step.

        Parameters
        ----------
        force_vec: numpy.ndarray
            The numpy.ndarray object of the force acting on the chain-of-state;  
            Should have shape chain.replica_vec.shape = (nrep, ndof).
            - nrep is the number of string images;
            - ndof is the degrees of freedom of each string image,
              For string images of N atoms in cartersian space, ndof is 3N.
            Note that force_vec is the negative of the gradient. 

        reparametrize: bool, default: True
            If reparametrization should be done after envolving the string.
            Note that in cases that reparametrization hasn't been done for this instance,  
            reparametrize would be done first.  
        """
        if not self._init_reparam: # If no parametrization has been done for the string.
            self._reparametrize()

        # get the projection of force vectors perpendicular to the string tangent.
        tang_vec = self._get_string_tangent()
        proj_force_vec = self._get_projected_force(force_vec, tang_vec)

        # evolve the string according to the projected force vectors.
        self.chain.replicas_vec = self._optimizer.evolve(self.chain.replicas_vec, proj_force_vec)
        
        if reparametrize:
            self._reparametrize()

class StrmInterpolatorBase(abc.ABC):
    """The abstract base class of interpolators for the string method."""

    @abc.abstractmethod
    def fit(self, x:numpy.ndarray, y:numpy.ndarray):
        """Fit the interpolator for points (x, y).

        Parameters
        ----------
        x: numpy.ndarray
            1-D values on the x axis.
        y: numpy.ndarray
            1-D values on the y axis.
        """

    @abc.abstractmethod
    def transform(self, x:numpy.ndarray):
        """Get new values from the interpolator. 
        CubicSplineInterpolator.fit(x, y) must be called before this method.
        
        Parameters
        ----------
        x: numpy.ndarray, 
            1-D values on the x axis to be evaluated. 
        """

    @abc.abstractmethod
    def get_gradient(self, x:numpy.ndarray):
        """Get the tangents at specified points.
        
        Parameters
        ----------
        x: numpy.ndarray, 
            1-D values on the x axis where the tangents are to be evaluated. 
        """

class StrmCubicSplineInterpolator(StrmInterpolatorBase):
    """The cubic spline interpolators for the string method."""

    def __init__(self):
        self._is_fitted = False

    def fit(self, x:numpy.ndarray, y:numpy.ndarray):
        """Fit the interpolator for points (x, y).

        Parameters
        ----------
        x: numpy.ndarray
            1-D values on the x axis.
        y: numpy.ndarray
            1-D values on the y axis.
        """
        self._intpol    = CubicSpline(x, y)
        self._is_fitted = True
        return self

    def transform(self, x:numpy.ndarray):
        """Get new values from the interpolator. 
        CubicSplineInterpolator.fit(x, y) must be called before this method.
        
        Parameters
        ----------
        x: numpy.ndarray, 
            1-D values on the x axis to be evaluated. 
        """
        if self._is_fitted == False:
            raise ValueError("The intepolator has not yet been fitted.")
        
        elif x.ndim != 1:
            raise ValueError("Interpolation must be performed on 1-D array.")

        else:
            return self._intpol(x)
    
    def get_gradient(self, x:numpy.ndarray):
        """Get the tangents at specified points.
        
        Parameters
        ----------
        x: numpy.ndarray, 
            1-D values on the x axis where the tangents are to be evaluated. 
        """
        if self._is_fitted == False:
            raise ValueError("The intepolator has not yet been fitted.")

        elif x.ndim != 1:
            raise ValueError("Interpolation must be performed on 1-D array.")
        
        else:   # cubic spline interpolator has analytical gradients. 
            return self._intpol(x, 1)