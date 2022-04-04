"""Implements the replica path method with holonomic RMS constraints.
Note: for some reason, the solver for the Lagrangion Multipliers could not solve 
      lambdas within ~30 steps as claimed by the original publication below,

        J.B. Brokaw, K.R. Haas, J.-W. Chu J. Chem. Theory Comput., 5, 8, 2009. 
        DOI:10.1021/ct9001398
    
     I probably need a better solver. 

Authors: Zilin Song.
"""
import scipy.sparse.linalg
import numpy
from cos.optimizers import OptimizerBase, SteepestDecent, AdaptiveMomentum
from replica.chain import ReplicaChainBase

class HolonomicPathMethod(object):
    """Class for managing the chain of replicas using the Holonomic Path method.
    
    Parameters
    ----------
    chain: ReplicaChainBase
        The chain of replicas.

    optimizer: str or a class instance that inherits OptimizerBase, default: 'sd'
        The optimization method for string evolution. Allowed values are
            "sd":   steepest decent optimizer;
            "adam": adaptive momentum optimizer.

    threshold: float, default: 10**-8
        The tolerance under which the holonomic constraint is considered converged.

    Attributes
    ----------
    chain: ReplicaChainBase
        The chain of replicas.

    nrep: int
        The number of replicas along the chain. 

    ndof: int
        The number of degrees of freedom in each replica.

    """
    def __init__(self, chain:ReplicaChainBase, optimizer:str='sd', threshold:float=10**-6):
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

    def evolve(self, force_vec:numpy.ndarray, patience:int=500, verbose:bool=True):
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
        
        patience: int, default:500
            The number of iteration steps the solver is allowed to solve for the Lagrangian Multipliers. 
        
        verbose: bool, default:True
            If the information of solving holonomic constraints should be echoed. 
        """
        # Optmization step.
        self.chain.replicas_vec = self._optimizer.evolve(self.chain.replicas_vec, force_vec)

        # ===========================================================================
        # ====================== Impose holonomic constraints. ======================
        # ========== solve lambdas for holonomic constraints of equal RMS. ==========
        # ===========================================================================

        # 0. Initialize

        ## (n-1)-th step.
        rep_a = self.chain.replicas_vec  # all replicas
        rms_a = self.chain.rms()         # initial rms.

        ##   (n)-th step.
        rep_b = numpy.copy(rep_a)
        rms_b = numpy.copy(rms_a)

        ## Control variables. 
        rms_mean    = numpy.mean(rms_b)    # the mean rms against which the holonomic constraints are imposed. 
        n_lambda    = self.nrep-1          # no. lambdas.        
        rms_maxdiff = 999.                 # convergence thresholds.
        step        = 0                    # step counter.

        # 1. iteratively solve for lambdas and coordinates for the correct update. 
        while rms_maxdiff >= self.threshold and step <= patience:
            # i.  build the lambda coefficient matrix, note that this matrix is tridiagonal
            lambda_coeff = numpy.zeros((n_lambda, n_lambda))

            for i in range(n_lambda):       # the i-th row

                lambda_coeff[i][i]       =  2. * numpy.sum((rep_b[i]-rep_b[i+1]) * (rep_a[i]   - rep_a[i+1])) / rms_a[i]   / rms_b[i]
                
                if i != 0:              # not the first row.
                    lambda_coeff[i][i-1] =  1. * numpy.sum((rep_b[i]-rep_b[i+1]) * (rep_a[i-1] - rep_a[i]  )) / rms_a[i-1] / rms_b[i]
                if i != n_lambda-1:     # not the  last row.
                    lambda_coeff[i][i+1] = -1. * numpy.sum((rep_b[i]-rep_b[i+1]) * (rep_a[i+1] - rep_a[i+2])) / rms_a[i+1] / rms_b[i]

            # ii.  solve for lambda.
            # solve for lambdas:  lambda_coeff * lambdas = mean-rms
            lambda_val = scipy.sparse.linalg.lsqr(lambda_coeff, rms_mean-rms_b)[0]

            # iii. update coordinates and rms from (n-1) to (n)
            rep_a = numpy.copy(rep_b)
            rms_a = numpy.copy(rms_b)

            # iv.  update coordinates and rms from (n) to (n+1)
            ## one could only truely fix the first and the last replicas by zeroing this update, but this is not implemented. 
            ## Note that zeroing out force vectors acting on the replicas will not prevent this part from updating coordinates. 
            for i in range(1, n_lambda):
                rep_b[i] = rep_a[i] + lambda_val[i-1] / rms_a[i-1] * (rep_a[i] - rep_a[i-1]) + lambda_val[i]/ rms_a[i] * (rep_a[i] - rep_a[i+1])

            rms_b = numpy.power(numpy.sum(numpy.power(rep_b[0:-1, :]-rep_b[1:, :], 2.), axis=1), 1/2.)
            
            # v.   convergence thresholds and iteration info.
            rms_maxdiff = numpy.max(numpy.abs(rms_b-rms_mean))
            step += 1

        # vi.  possibly verbose. 
        if verbose:
            print(f"step: {step} {rms_maxdiff}")

        # ===========================================================================
        # ================================== Done. ==================================
        # ===========================================================================

        # Update chain-of-states
        self.chain.replicas_vec = rep_b
