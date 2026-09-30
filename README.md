# Jetstream2 AI Fellow Project
This repo hosts a collection of tutorials and codebases for my Jetstream2 AI Fellow Project - "Inverse Design of Crystal Structures with JAX-MD".

## What are the goals for this project?
For this project, we have two major goals:
1. provide a collection of tutorials to help other users use JAX-MD using Jetstream2
2. optimize patchy particle parameters for robust crystal self-assembly 

## What is JAX-MD?
JAX-MD is an end-to-end differentiable molecular dynamics (MD) engine that makes parameter optimizations through a MD simulation easy. For more information, check out [JAX-MD's GitHub Repo](https://github.com/jax-md/jax-md). 

## Table of Content
[JAX-MD Installation Tutorial for Jetstream2](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/installation.md)

[Helper Function Library for Lattice Initialization and Loss Function](https://github.com/xiyudu/jetstream2_ai_fellow/tree/main/helper_function)

Tutorial Notebooks:
1. [2D Forward Assembly](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/2D_forward_lj.ipynb)
2. [2D Forward Assembly with Patchy Particle](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/2D_forward_square.ipynb)
3. [3D Forward Assembly with Patchy Particle](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/3D_forward_cubic.ipynb)
4. [2D Square Lattice Optimization Demo](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/2D_square_opt_demo.ipynb)

We also provide sample code for various [structure initialization](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/helper_function/structures_init.ipynb) and optimization script for designing [Kagome](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/2D_kagome_opt_run.py) and [Diamond structures](https://github.com/xiyudu/jetstream2_ai_fellow/blob/main/tutorial/3D_diamond_opt_run.py).
