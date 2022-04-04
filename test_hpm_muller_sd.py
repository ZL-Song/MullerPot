"""A test for string method on 2D Muller potential.

Authors: Zilin Song.
"""

from cos.hpm import HolonomicPathMethod
from replica.chain import Replica2DChain
from potentials.pot_2d import MullerPotential
import numpy
import matplotlib.pyplot as plt

# Define the initial chain-of-states.
nrep = 20
replicas = numpy.zeros((nrep, 2))
replicas[:, 0] = 0
replicas[:, 1] = numpy.linspace(0, 2, num=nrep)
# replicas[:, 0] = numpy.linspace(0.62318998, -0.55879296, num=nrep)
# replicas[:, 1] = numpy.linspace(0.02797983,  1.44115597, num=nrep)

# 2D chain-of-states.
chain_2d = Replica2DChain(replicas)

# 2D Muller potential.
mbp = MullerPotential()

# Initial plot
fig, ax = mbp.plot_contourf()
ax.scatter(chain_2d.cartersian_coors[:, 0], chain_2d.cartersian_coors[:, 1], s=0.4, c='k', zorder=5)

# Optimizer & Interpolator

# String method.
cos_hpm = HolonomicPathMethod(chain_2d)

# Iterate for 150 steps.
niter=150
for _ in range(niter):
    force_ = mbp.get_force(cos_hpm.chain.cartersian_coors)
    cos_hpm.evolve(force_)

    ax.plot(cos_hpm.chain.cartersian_coors[:, 0], cos_hpm.chain.cartersian_coors[:, 1], 'o-', lw=0.5, markersize=0.5, c='grey' )


ax.scatter(cos_hpm.chain.cartersian_coors[:, 0], cos_hpm.chain.cartersian_coors[:, 1], s=0.5, c='r', zorder=5)

plt.savefig('test_hpm_muller_sd.png')
