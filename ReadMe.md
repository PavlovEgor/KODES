# Kinetic Ordinary Differential Equations Solver (KODES)

A library for solving multiple systems of ordinary differential equations on GPUs.

For CFD calculations with chemical reactions, due to the significant difference in the time scales of chemical and hydrodynamic processes, it is customary to separate chemical reactions into a separate step and solve them as ordinary differential equations (ODEs) in each cell of the computational grid. In this case, systems from neighboring cells do not affect each other, which allows for direct parallelization of the entire calculation process.

The approach of parallelizing the solution of the ODE system itself (for example, in SUNDUALS, on which Cantera is based) is quite limited, as its efficiency should increase with the size of the vectors, which in chemical kinetics are the unknown concentrations of the mixture components. However, the number of components is usually determined not by the user in the CFD calculation, but by the creator of the kinetic mechanisms, making it difficult to scale.

Unlike the classical approach, each system will be solved on a small number of CUDA cores, while multiple systems will be solved simultaneously, significantly increasing the calculation speed for regions with a large number of cells.
