"""A test for string method on 2D Muller potential.

Authors: Zilin Song.
"""

from cos.stringm import StringMethod
from cos.optimizers import SteepestDecent
from potentials.pot_2d import MullerPotential
from replica.chain import Replica2DChain
import numpy
import matplotlib.pyplot as plt

# Define the initial chain-of-states.
nrep = 20
replicas = numpy.zeros((nrep, 2))
replicas[:, 0] = 0
replicas[:, 1] = numpy.linspace(0, 2, num=nrep)

# 2D chain-of-states.
chain_2d = Replica2DChain(replicas)

# 2D Muller potential.
mbp = MullerPotential()

# Initial plot
fig, ax = mbp.plot_contourf()
ax.scatter(chain_2d.cartersian_coors[:, 0], chain_2d.cartersian_coors[:, 1], s=0.4, c='k', zorder=5)

# Optimizer.
opt = SteepestDecent()

# String method.
cos_stringm = StringMethod(chain_2d, optimizer=opt, intpol_method='cspline')

# Iterate for 150 steps.
niter=150
for _ in range(niter):
    force_ = mbp.get_force(cos_stringm.chain.cartersian_coors)
    cos_stringm.evolve(force_, reparametrize=True)

    ax.plot(cos_stringm.chain.cartersian_coors[:, 0], cos_stringm.chain.cartersian_coors[:, 1], 'o-', lw=0.5, markersize=0.5, c='grey' )

cos_stringm._reparametrize()
ax.scatter(cos_stringm.chain.cartersian_coors[:, 0], cos_stringm.chain.cartersian_coors[:, 1], s=1, c='r', zorder=5)

plt.savefig('test_stringm_muller_sd.png')