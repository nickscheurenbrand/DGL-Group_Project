# DGL-Group_Project
# DGL2026 Brain Graph Super-Resolution Challenge

## Contributors

- Temi Messmer
- Daniel Sanchez Sanchez
- Nicholas Scheurenbrand
- Vikneswarean Indiran

## Problem Description

The objective is to design a generative Graph Neural Network (GNN) trained in an inductive setting to predict high-resolution (HR) brain connectivity graphs from low-resolution (LR) counterparts. The data is represented as symmetric weighted connectivity matrices where each element quantifies the neural correlation between two distinct brain regions. 
Given an LR connectivity matrix $\mathbf{A}^{LR} \in \mathbb{R}^{160 \times 160}$, the model must generate the corresponding HR connectivity matrix $\mathbf{A}^{HR} \in \mathbb{R}^{268 \times 268}$ of the same brain. Mathematically, the goal is to learn a mapping $f$ such that: $$f(\mathbf{A}^{LR}) = \hat{\mathbf{A}}^{HR} \approx \mathbf{A}^{HR}$$ where $\hat{\mathbf{A}}^{HR}$ is the adjacency matrix predicted by the GNN model $f$.

Enhancing brain graph resolution computationally offers significant benefits for neuroscience and clinical imaging. Obtaining high-resolution brain scans is often expensive and time-consuming. By framing this as a graph super-resolution problem, researchers can artificially upgrade legacy or lower-quality clinical data to extract fine-grained neural insights without requiring new physical scans. Furthermore, this challenge pushes the boundaries of geometric deep learning. It requires the model to learn complex topological structures and generalize to unseen brain graphs in an inductive setting.

## Name of your model - Methodology

- Summarize in a few sentences the building blocks of your generative GNN model.

- Figure of your model.

## Used External Libraries

- Give instructions on how to install the external libraries if you have used any in your code.

## Results

- Insert your bar plots.


## References

- Do not forget to include the references to methods you used to build your model.
