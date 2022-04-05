# MullerPot
My playground with the Muller potential.

### Chain-of-state methods for finding MEPs

- ./cos/stringm.py
> Implemented the String Method (with force projections) on the Muller potential.  
> Reference:  
>   String Method for the Study of Rare Events.  
>   W. E, E. Ren, E. Vanden-Eijnden, Phys. Rev. B 66, 052301  
>   DOI:10.1103/PhysRevB.66.052301  

- ./cos/hpm.py
> Implemented the Holonomic Path Method on the Muller potential.  
> Reference:  
>   Reaction Path Optimization with Holonomic Constraints and Kinetic Energy Potentials.  
>   J.B. Brokaw, K.R. Haas, J.-W. Chu, J. Chem. Theory Comput., 5, 8, 2009.  
>   DOI:10.1021/ct9001398  
> 
> In some of my papers, this method was called the "Replica Path Method with holonomic constraints".  
> This is not the offical name as no official name for this method was proposed by the authors.  
> I'll just call it Holonomic Path Method for now.  

### Some Jupyter notebooks related to the String Method:
- ./examples/0ksm_muller  
> 0ksm_muller/stringm_muller.ipynb:  String Method on a Muller potential surface, without using Numpy;  
- ./examples/voro_fe  
> voro_fe/hwfe_voro_sm.ipynb:  Used to post-processes the Voronoi (hard wall) collision matrix from CHARMM string method simulations, which gives MFEP profiles;  
> voro_fe/voro0.dat:  Data file (of the Voronoi HW simulation ALA dipeptide dihedral rotation) used as an example in hwfe_voro_sm.ipynb.  
