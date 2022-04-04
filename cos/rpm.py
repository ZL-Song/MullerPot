"""Implements the replica path methods.

Authors: Zilin Song.
"""
import scipy.sparse.linalg
import numpy
from cos.optimizers import OptimizerBase, SteepestDecent, AdaptiveMomentum
from replica.chain import ReplicaChainBase

class ReplicaPathMethod(object):
    """Class for managing the chain of replicas using the string method.
    
    Parameters
    ----------
    chain: ReplicaChainBase
        The chain of replicas.

    optimizer: str or a class instance that inherits OptimizerBase, default: 'sd'
        The optimization method for string evolution. Allowed values are
            "sd":   steepest decent optimizer;
            "adam": adaptive momentum optimizer.
    threshold: float, default: 10**-8
        The threshold under which the holonomic constraint is considered converged.

    Attributes
    ----------
    chain: ReplicaChainBase
        The chain of replicas.

    nrep: int
        The number of replicas along the chain. 

    ndof: int
        The number of degrees of freedom in each replica.

    """
    def __init__(self, chain:ReplicaChainBase, optimizer:str='sd', threshold:float=10**-8):
        self.chain = chain                              # Chain-of-States
        self.nrep  = self.chain.replicas_vec.shape[0]   # No. replicas.
        self.ndof  = self.chain.replicas_vec.shape[1]   # No. degrees of freedom.
        self.threshold = threshold                      # Threshold for the convergence of holonomic constraints.

        # Determine optimizer
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

    def _enforce_equal_rms(self):
        """Redistribute replicas for equal rms distances."""
        # (n-1)-th step.
        rep_a = self.chain.replicas_vec  # all replicas
        rms_a = self.chain.rms()         # initial rms.

        #   (n)-th step.
        rep_b = numpy.copy(rep_a)  # all replicas

        # get the mean rms for holonomic contraints
        rms_mean = numpy.mean(self.chain.rms())  # averaged rms distances between adjacent replicas

        # number of lambda to solve
        n_lambda = self.nrep-1

        max_diff = 999.
        max_lambda = 999.
        steps    = 0
        while max_diff >= self.threshold and steps <= 100 and max_lambda >= 10**-5:
            
            lambda_coeff = numpy.zeros((n_lambda, n_lambda))

            rms = numpy.power(numpy.sum(numpy.power(rep_b[0:-1, :]-rep_b[1:, :], 2.), axis=1), 1/2.)
            y = numpy.mean(rms_mean) - rms

            max_diff = numpy.max(numpy.abs(y))
            steps += 1
            # make the matrix for solving lambda.
            # i is rows
            for i in range(n_lambda):

                lambda_coeff[i][i] = 2. * numpy.sum((rep_b[i]-rep_b[i+1]) * (rep_a[i] - rep_a[i+1])) / rms_a[i]   / rms[i] 
                
                if i != 0:
                    lambda_coeff[i][i-1] = -1. * numpy.sum((rep_b[i]-rep_b[i+1]) * (rep_a[i]   - rep_a[i-1])) / rms_a[i-1] / rms[i]
                if i != n_lambda-1:
                    lambda_coeff[i][i+1] = -1. * numpy.sum((rep_b[i]-rep_b[i+1]) * (rep_a[i+1] - rep_a[i+2])) / rms_a[i+1] / rms[i]

            # solve for lambdas:  lambda_coeff * lambdas = y
            lambda_val = scipy.sparse.linalg.lsqr(lambda_coeff, y)[0]
            max_lambda = numpy.max(numpy.abs(lambda_val))
            # update coordaintes acoordingly.
            ## update (n-1) to n
            rep_a = numpy.copy(rep_b)
            rms_a = numpy.copy(rms)
            # fix the first and the last -> this is not necessary.
            for i in range(1, n_lambda):
                rep_b[i] = rep_a[i] + lambda_val[i-1] / rms_a[i-1] * (rep_a[i] - rep_a[i-1]) + lambda_val[i]/ rms_a[i] * (rep_a[i] - rep_a[i+1])

            print("steps: ", steps, rms, lambda_val)
        return rep_b

    def _lambda_coef(self):
        """build and return the lambda coefficient matrix."""

    def evolve(self, force_vec:numpy.ndarray):
        """Evolve the chain-of-replicas using one optmizer step and re-distribute for equal RMS distances.
        
        Parameters
        ----------
        force_vec: numpy.ndarray
            The numpy.ndarray object of the force acting on the chain-of-state;  
            Should have shape chain.replica_vec.shape = (nrep, ndof).
            - nrep is the number of string images;
            - ndof is the degrees of freedom of each string image,
              For string images of N atoms in cartersian space, ndof is 3N.
            Note that force_vec is the negative of the gradient. 
        """
        # fix the first and the last.
        #force_vec[ 0, :] = numpy.zeros((self.ndof))
        #force_vec[-1, :] = numpy.zeros((self.ndof))

        # optmization step.
        self.chain.replicas_vec = self._optimizer.evolve(self.chain.replicas_vec, force_vec)

        self.chain.replicas_vec = self._enforce_equal_rms()
        # solve lambdas for holonomic constraints of equal RMS.
        # 1. Build lambda coefficient matrix.
        
        
        # 2. Solve for lambda.


        # 3. Update coordinates.