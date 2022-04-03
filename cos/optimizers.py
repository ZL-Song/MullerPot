"""Implements all types of optimizers.

Authors: Zilin Song.
"""

import numpy, abc

class OptimizerBase(abc.ABC):
    """The abstract base class for optimizers.
    All classes inherits this class should implement the evolve(self, x, force) method. 
    """
    @abc.abstractmethod
    def evolve(self, x:numpy.ndarray, force:numpy.ndarray):
        """Execulte one steepest decent step.
        
        Parameters
        ----------
        x: numpy.ndarray
            The replica array.

        force: numpy.ndarray
            The force array.
        """

class SteepestDecent(OptimizerBase):
    """The Steepest Decent (SD) optimizer with optional adaptive optimization step-sizes.

    Parameters
    ----------
    eta: float, default=0.0002
         The step size for steepest decent optimization step.

    ada_eta: float, default=1
         Adapt eta values at each call to evolve() according to:
            eta_{t+1} = eta_{t} * ada_eta.
         Recommended values are 0.90 ~ 1.00.
         Default value 1 means no adaptive eta;
         ada_eta<1 means decreasing step size;
         ada_eta>1 means increasing step size (not recommended).
    """

    def __init__(self, eta:float=0.0002, ada_eta:float=1.):
        self._eta     = eta
        self._ada_eta = ada_eta
    
    def evolve(self, x:numpy.ndarray, force:numpy.ndarray):
        """Execulte one steepest decent step.
        
        Parameters
        ----------
        x: numpy.ndarray
            The replica array.

        force: numpy.ndarray
            The force array.
        """
        x += self._eta * force       # force = -grad

        self._eta *= self._ada_eta

        return x

class AdaptiveMomentum(OptimizerBase):
    """The Adaptive Momentum (AdaM) optimizer.
    
    Parameters
    ----------
    eta: float, default=0.0002
         The step size for steepest decent optimization step.

    adam_beta1: float, default=.9
         The AdaM parameter beta_{1}.

    adam_beta2: float, default=.999
         The AdaM parameter beta_{2}.

    adam_epsilon: float, default=1e-8
         The AdaM parameter epsilon.
    """
    def __init__(self, eta:float=0.01, adam_beta1:float=.9, adam_beta2:float=.999, adam_epsilon:float=1e-8):
        self._eta           = eta
        self._adam_beta1    = adam_beta1
        self._adam_beta2    = adam_beta2
        self._adam_epsilon  = adam_epsilon
        self._adam_t        = 0
    
    def evolve(self, x:numpy.ndarray, force:numpy.ndarray):
        """Execulte one AdaM step.
        
        Parameters
        ----------
        x: numpy.ndarray
            The replica array.

        force: numpy.ndarray
            The force array.
        """
        if self._adam_t == 0:  # initialize M and V at initial step.
            self._adam_M = 0
            self._adam_V = 0

        self._adam_t += 1

        # step1: M^{t+1} = beta_1 * M^{t} + (1-beta_1) * grad;      grad = -force;
        self._adam_M = self._adam_beta1 * self._adam_M - (1-self._adam_beta1) * force

        # step2: V^{t+1} = beta_2 * V^{t} + (1-beta_2) * grad**2;   grad**2 = force**2;
        self._adam_V = self._adam_beta2 * self._adam_V + (1-self._adam_beta2) * force**2

        # step3: M^{hat,t+1} = M^{t+1} / (1 - beta_1**t);
        adam_M_hat = self._adam_M / (1 - self._adam_beta1**self._adam_t)

        # step4: V^{hat,t+1} = V^{t+1} / (1 - beta_2**t);
        adam_V_hat = self._adam_V / (1 - self._adam_beta2**self._adam_t)

        # step5: x^(t+1) = x^{t} - eta * M^{hat,t+1} / (V^{hat,t+1} ** 0.5 + epsilon); grad = -force;
        x -= self._eta * adam_M_hat / (adam_V_hat**0.5 + self._adam_epsilon)

        return x

# TODO: code more optimizers.