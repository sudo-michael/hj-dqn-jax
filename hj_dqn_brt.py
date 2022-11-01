# %%
import os
# JAX will preallocate 90% of currently-available GPU memory when the first JAX operation is run, let's not
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from flax.training.train_state import TrainState
import flax.linen as nn
import optax
import numpy as np
import argparse
import matplotlib.pyplot as plt
import wandb
import tqdm
import datetime

# %%
def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--exp-name", type=str, required=True,
        help="name of experiment")
    parser.add_argument("--notes", type=str, required=False,
        help="note")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-epochs", type=int, default=50_000,
        help="epochs")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=64_000,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-g", type=float, default=0.9,
        help="the starting gamma")
    parser.add_argument("--end-g", type=float, default=0.99999,
        help="the ending gamma")
    # fmt: on
    args = parser.parse_args()
    return args


# %%
# nn.Module is a DataClass
class QNetwork(nn.Module):
    action_dim: int


    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        x = theta_to_cos_sin(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict

@jax.vmap
def theta_to_cos_sin(x: jnp.array) -> jnp.array:
    new_x = jnp.array([x[0], x[1], jnp.cos(x[2]), jnp.sin(x[2])])
    return new_x

def linear_schedule(start_g: float, end_g: float, duration: int, t: int) -> float:
    assert start_g <= end_g
    slope = (end_g - start_g) / duration
    return min(slope * t + start_g, end_g)

    # import matplotlib.pyplot as plt
    # import numpy as np

    # start_g = 0.9
    # end_g = 0.99
    # schedule = linear_schedule(start_g, end_g, 50, np.arange(100))
    # plt.plot(np.arange(100), schedule)

def sample_dataset(batch_size: int, key) -> jnp.array:
    low = jnp.array([-1.0, -1.0, -jnp.pi])
    high = jnp.array([1.0, 1.0, jnp.pi])
    return jax.random.uniform(key, (batch_size, 3), minval=low, maxval=high)


brt = jnp.array(np.load("./brt.npy"))

def odp_periodic_linspace(start: float, stop: float, num=50) -> np.ndarray:
    min = start
    max = start + (stop - start) / (1 - 1/num)
    return np.linspace(min, max, num)

def eval(q_state, epoch):
    grid_points = 101
    grid_space = jnp.linspace(
        0,
        grid_points,
        num=grid_points,
        endpoint=False,
    )
    grid_coords = jnp.dstack(
        jnp.meshgrid(grid_space, grid_space, indexing="ij")
    ).reshape(
        -1, 2
    )  # (grid_points**2, 2)
    # scale to [-1, 1]
    grid_coords /= grid_points - 1
    grid_coords -= 0.5
    grid_coords *= 2

    pi = odp_periodic_linspace(-np.pi, np.pi, grid_points)

    # create a batch for all points in saved brt
    big_grid_coords = jnp.tile(grid_coords, (grid_points, 1))  # (grid_points** 2 * 41, 2)
    pis = jnp.tile(pi, (grid_points**2, 1)) # (grid_points**2,  len(pis))
    # stack each column on top of each other
    # [[-1, 0, 1],      [[-1],
    #  [-1, 0, 1]] --->  [-1],
    #                    [ 0],
    #                     ...
    pis = pis.T.reshape(-1, 1)  #  (grid_points**2 * len(pis), 1)
    big_grid = jnp.hstack((big_grid_coords, pis))  # (grid_points**2 * len(pis), 3)
    q_value = q_state.apply_fn(q_state.params, big_grid)
    values = q_value.max(axis=-1)  # grid_points**2 * len(pis)
    # the order of big_grid can be though of stacking (grid_points, pi) on top of each of each other
    # so the first (grid_points**2) rows are for pi[0]
    # so the second (grid_points**2) rows are for pi[1]
    # this is why we must reshape in (41, grid_points, grid_points)
    # since the first axis represents pi
    predicted_brt = jnp.moveaxis(values.reshape((grid_points, grid_points, grid_points)), 0, -1)
    error = ((predicted_brt - brt) ** 2).mean()
    wandb.log({"mse_error": error}, step=epoch)

    slices = [0, 10, 20, 30, 50, 60]
    axies_coords = [(0, 0), (1, 0), (0, 1), (1, 1)]
    fig, axis = plt.subplots(6, 2, figsize=(10, 50))
    for i, slice in enumerate(slices):
        value = predicted_brt[:, :, slice]
        axis[(i, 0)].set_title(f"theta={pi[slice]/np.pi * 180:0.2f}")
        axis[(i, 0)].imshow(
            (value > 0).T, cmap="bwr", origin="lower", extent=(-1.0, 1.0, -1, 1.0)
        )
        axis[(i, 1)].set_title(f"[odp] theta={pi[slice]/np.pi * 180:0.2f}")
        axis[(i, 1)].imshow(
            (brt[:, :, slice] > 0).T, cmap="bwr", origin="lower", extent=(-1.0, 1.0, -1, 1.0)
        )
    wandb.log({"brt": fig,
               }, step=epoch)
    

    plt.close(fig)

def dynamics(state: jnp.array, control: float):
    x1 = jnp.cos(state[2])
    x2 = jnp.sin(state[2])
    x3 = control
    return jnp.array([x1, x2, x3])

@jax.vmap
@jax.jit
def step(obs, actions):
    # obs: (3, )
    # actions: int,

    # {0, 1, 2} -> {-1, 0, 1}
    actions = actions - 1
    # {-1, 0, 1} -> {-1.5, 0, 1.5}
    actions = actions * 1.5

    next_obs = dynamics(obs, actions) * 0.05 + obs
    
    lx = jnp.linalg.norm(next_obs[:2]) - 0.5
            
    return next_obs, lx


if __name__ in "__main__":
    args = get_args()
    wandb.init(project="hj_dqn", config=args, notes=args.notes, save_code=True)
    run_name = f"{args.exp_name}_{datetime.datetime.today().strftime('%b%d_%H_%M')}"

    q_network = QNetwork(action_dim=3)

    key = jax.random.PRNGKey(0)
    key, q_key = jax.random.split(key)
    obs = jnp.ones((1, 3))

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=3e-4),
    )

    @jax.jit
    def update(q_state, gamma, obs, actions, next_obs, rewards):
        q_next_target = q_network.apply(
            q_state.target_params, next_obs
        )  # (b, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (b, )
        next_q_value = (1 - gamma) * rewards + gamma * jnp.minimum(
            rewards, q_next_target
        )  # (b, )

        def mse_loss(params):
            q_pred = q_network.apply(params, obs)  # (b, num_actions)
            q_pred = q_pred[np.arange(q_pred.shape[0]), actions]  # (b, )
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params
        )
        q_state = q_state.apply_gradients(grads=grads)

        return q_state, q_pred, loss_value

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        gamma = linear_schedule(args.start_g, args.end_g, int(args.num_epochs * 3/4), epoch)
        key, obs_key = jax.random.split(key)
        obs = sample_dataset(args.batch_size, obs_key)
        q_values = q_state.apply_fn(q_state.params, obs)
        actions = q_values.argmax(axis=-1)  # (b, )
        next_obs, rewards = step(obs, actions)  # (b, 3), (b, )

        q_state, q_pred, loss = update(q_state, gamma, obs, actions, next_obs, rewards)
        wandb.log({"loss": loss, "epoch": epoch}, step=epoch)

        if (epoch + 1) % 1000 == 0:
            eval(q_state, epoch)
            checkpoints.save_checkpoint(
                ckpt_dir="ckpts",
                target=q_state,
                step=epoch,
                overwrite=True,
                prefix=f"checkpoint_brt_{run_name}_",
            )
            print(f"{epoch=} {loss=}")

        q_state = q_state.replace(
            target_params=optax.incremental_update(
                q_state.params, q_state.target_params, 0.005
            )
        )

