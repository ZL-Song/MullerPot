"""Implements some 2D potentials and corresponding analytical gradients.

Authors: Zilin Song.
"""

import numpy
import matplotlib.pyplot as plt

class MullerPotential(object):
    """The 2D Müller-Brown potential."""
    
    def __init__(self):
        # M-B potential constants
        self.AA = [-200., -100.,  -170.,  15., ]
        self.aa = [  -1.,   -1.,   -6.5,  0.7, ]
        self.bb = [   0.,    0.,    11.,  0.6, ]
        self.cc = [ -10.,  -10.,   -6.5,  0.7, ] 
        self.xx = [   1.,    0.,   -0.5,  -1., ]
        self.yy = [   0.,   0.5,    1.5,   1., ]

    def get_energy(self, coordinates:numpy.ndarray):
        """Return the energies of the input 2D points x.

        Parameters
        ----------
        coordinates: numpy.ndarray
            The input points (x, y), note that the shape of coordinates must be (-1, 2).
        """

        if coordinates.shape[1] != 2:
            raise ValueError('Muller potential is 2D, thus the shape of x must be (-1, 2).')
        
        else:
            ene = 0.
            for i in range(4):
                ene += self.AA[i] * numpy.exp(
                    self.aa[i] * (coordinates[:, 0] - self.xx[i])**2 + 
                    self.bb[i] * (coordinates[:, 0] - self.xx[i]) * (coordinates[:, 1] - self.yy[i]) + 
                    self.cc[i] * (coordinates[:, 1] - self.yy[i])**2
                )

        return ene

    def get_force(self, coordinates:numpy.ndarray):
        """Return the forces (-gradients) of the input 2D points x.

        Parameters
        ----------
        coordinates: numpy.ndarray
            The input points (x, y), note that the shape of coordinates must be (-1, 2).
        """

        if coordinates.shape[1] != 2:
            raise ValueError('Muller potential is 2D, thus the shape of coordinates must be (-1, 2).')

        else:
            force = numpy.zeros(coordinates.shape)
            for i in range(4):
                u = self.AA[i] * numpy.exp(
                            self.aa[i] * (coordinates[:, 0] - self.xx[i])**2 +
                            self.bb[i] * (coordinates[:, 0] - self.xx[i]) * (coordinates[:, 1] - self.yy[i]) +
                            self.cc[i] * (coordinates[:, 1] - self.yy[i])**2 
                )

                # note that force must be negative gradient
                force[:, 0] -= (u * (self.aa[i]*2*(coordinates[:, 0] - self.xx[i]) + self.bb[i]*(coordinates[:, 1] - self.yy[i])))
                force[:, 1] -= (u * (self.cc[i]*2*(coordinates[:, 1] - self.yy[i]) + self.bb[i]*(coordinates[:, 0] - self.xx[i])))

            return force
    
    def plot_contourf(self, minx=-1.5, maxx=1.3, miny=-0.5, maxy=2.3, xpoints=101, ypoints=101):
        """Plot the Müller-Brown potential as a contour map.
        
        Parameters
        ----------
        minx, maxx, miny, maxy: float
            The range of potential surface.
        xpoints, ypoints: int
            The number of grid points on each dimension.
            Must be positive.
            
        Returns
        -------
        fig, ax: 
            The canvas of the contourf plot.
        """
        # build the x-y grid dimensions
        # reinvent the np.linspace()
        x_coor = numpy.linspace(minx, maxx, num=xpoints)
        y_coor = numpy.linspace(miny, maxy, num=ypoints)

        xx, yy = numpy.mgrid[minx:maxx:xpoints*1j, miny:maxy:ypoints*1j]
        coor  = numpy.vstack([xx.ravel(), yy.ravel()]).T

        v = self.get_energy(coor)
        v = numpy.where(v <= 200, v, 200)

        fig, ax = plt.subplots(figsize=(6,6), dpi=300)
        ax.contourf(x_coor, y_coor, v.reshape((xpoints, ypoints)).T, 200)

        return fig, ax
