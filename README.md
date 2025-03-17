# multi-phase-sbm

## Description
Collection of codes for the multiphase-smoothed-boundary method and the corresponding publication "Simulation of intercalation and phase transitions in nano-porous, polycrystalline agglomerates"

The abstract reads:

 > Optimal microstructure design of battery materials is critical to enhance the performance of batteries
for tailored applications such as high power cells. Accurate simulation of the thermodynamics,
transport, and electrochemical reaction kinetics in commonly used polycrystalline battery materials
remains a challenge. Here, we combine state-of-the-art multiphase field modelling with the
smoothed boundary method to accurately simulate complex battery microstructures and multiphase
physics. The phase-field method is employed to parameterize complex open pore cathode
microstructures and we present a formulation to impose galvanostatic charging conditions on the
diffuse boundary representation. By extending the smoothed boundary method to the multiphasefield
method, we build a simulation framework which is capable of simulating the coupled effects
of intercalation, anisotropic diffusion, and phase transitions in arbitrary complex polycrystalline
agglomerates. This method is directly compatible with voxel-based data, e.g. from X-ray tomography.
The simulation framework is used to study the reversible phase transitions in LiXNiO2 in
dense and nanoporous agglomerates. Based on the thermodynamic consistency of phase-field approaches
with ab-initio simulations and the open circuit potential, we reconstruct the Gibbs free
energies of four individual phases (H1, M, H2 and H3) from experimental cycling data. The results
show remarkable agreement with previously published DFT results. From charge simulations, we
discover a strong influence of particle morphology on the phase transition behaviour, in particular
a shrinking core-like behaviour in dense polycrystalline structures and a particle-by-particle
mosaic behavior in nanoporous samples. Overall, the proposed simulation framework enables the
detailed study of phase transitions in intercalation materials to enhance microstructure design and
fast charging protocols.

## Installation
The code is based on python including the following libraries
- numpy
- scipy
- matplotlib
- pyvista

## Author contributions
SD conceptualized the work and carried out the implementation and validation of the simulation code.
SD, MR, DS and QH all contributed to the formulation and validation of the MP-SBM method. SD,
AEC and MZB formulated the electrochemical model; simulation studies and data visualization were
carried out by SD and MW. DS and BN provided funding for the project. SD, AEC, QH, MZB and
BN wrote the manuscript. All authors read and approved the final manuscript.

## Acknowledgments
This work contributes to the research performed at CELEST (Center for Electrochemical Energy
Storage Ulm-Karlsruhe) and was funded by the German Research Foundation (DFG) under Project
ID 390874152 (POLiS Cluster of Excellence). Support by the Helmholtz association though the MTET
programme (no. 38.02.01) is gratefully acknowledged.

## License
This code has been published under the MIT licence.