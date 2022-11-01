# HJ DQN JAX
An implementation of "Bridging hamilton-jacobi safety analysis and reinforcement learning" by Jaime F Fisac, Neil F Lugovoy, Vicenç Rubies-Royo, Shromona Ghosh, Claire J Tomlin

For simplicity, states are just sampled uniformally. 

Example:
```
python hj_dqn_brt.py --num-epochs 10_000 -batch-size 16_000 --exp-name dubins3d
```
# Dependancies
* python=3.8
* jax
* optax
* flax
* wandb
* tqdm
* [optimized_dp](https://github.com/SFU-MARS/optimized_dp)