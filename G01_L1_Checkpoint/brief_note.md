# QXU6028 L1 checkpoint brief note - G01

At 300 K, I evaluated the same linear density of states and Fermi function with two lower integration limits: 0 eV for the band-edge count and the exact Fermi level for the Fermi-referenced count. The gate ratios are C(0)=1.000, C(0.073)=2.005, C(0.100)=2.599, and C(0.200)=5.104; thus the counts agree at neutrality and the correction grows monotonically with positive doping.

At the exact operating point EF=0.098691150 eV, C=2.568739. The corrected carrier density is 3.405251e+11 cm^-2 and the corrected mobility is 32173.604 cm^2 V^-1 s^-1. At fixed sheet resistivity, mobility is inversely proportional to carrier density, so replacing the overestimated band-edge density by n_F raises both mobility and the inferred sensor sensitivity by the same factor, 2.569 (about 156.9%). The device specification should therefore use the Fermi-referenced carrier count; otherwise sensitivity is understated by this factor.

Numerical integration was checked against the closed form for n_F at the operating point (relative difference 2.35e-09). The plotted C=2 and C=4 crossings are 0.072762 eV and 0.157347 eV, respectively.
