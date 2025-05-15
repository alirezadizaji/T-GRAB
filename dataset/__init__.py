"""
Dataset
===
This folder contains all processes needed to generate/prepare data for the training/inference.
The architecture of this directory looks as follows:
[`DTDG`: Discrete time dynamic graph]:
    [`graph generation`: all scripts needed to generate graph are located here.]
        [Graph generation type1: causality]
            memory_node.py: Pattern/discovery temporal graph generation.
            ...
        [Graph generation type2: periodicity]
            er_clique.py: Random graph generation via erdos-renyi. During weekdays, a clique is being attached to this random graph, and during weekends, the clique is being removed.
            ...
        [Graph generation type3]
            ...
[`CTDG`: Continuous time dynamic graph]
    ...
[`utils`: All necessary scripts that are shared among both DTDG and CTDG for utilization]
    negative_generator.py: This is an extension of TGB benchmark negative generator. This makes it possible to generative negative sampling
"""