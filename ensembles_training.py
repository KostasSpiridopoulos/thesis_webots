import hydra
import torch
from lightning.fabric import Fabric
from lightning.pytorch.utilities.seed import isolate_rng
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.models.models import MLP
from sheeprl.utils.utils import dotdict
from omegaconf import OmegaConf
import pathlib
from sheeprl.utils.env import make_env
from sheeprl.algos.dreamer_v3.agent import build_agent
import gymnasium as gym
import numpy as np
from torch import nn
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
import os
from sheeprl.utils.distribution import (
    MSEDistribution
)
from sheeprl.models.models import LayerNorm

# path of your checkpoint
ckpt_path = pathlib.Path(r"/home/kkspyrid/sheeprl/logs/runs/dreamer_v3/PongNoFrameskip-v4/2024-10-14_19-41-14_dreamer_v3_PongNoFrameskip-v4_30/version_0/checkpoint/ckpt_508000_0.ckpt")


seed = 12
fabric = Fabric(accelerator="cuda", devices=1)
fabric.launch()
model_state = fabric.load(ckpt_path)
cfg = dotdict(OmegaConf.to_container(OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True))
model_cfg = cfg.copy()
torch.set_float32_matmul_precision('medium')
# Environment setup
vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
envs = vectorized_env(
    [
        make_env(
            cfg,
            cfg.seed + 0 * cfg.env.num_envs + i,
            0 * cfg.env.num_envs,
            "./imagination",
            "imagination",
            vector_env_idx=i,
        )
        for i in range(cfg.env.num_envs)
    ]
)
action_space = envs.single_action_space
observation_space = envs.single_observation_space

is_continuous = isinstance(action_space, gym.spaces.Box)
is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
actions_dim = tuple(
    action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
)
clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r

# The number of environments is set to 1
#cfg.env.num_envs = 4

print(cfg.env.num_envs)

ens_list = []
cfg_ensembles = cfg.algo.ensembles
ensembles_ln_cls = hydra.utils.get_class("sheeprl.models.models.LayerNorm")
with isolate_rng():
    for i in range(8):
        fabric.seed_everything(cfg.seed + i)
        ens_list.append(
            MLP(
                input_dims=int(
                    sum(actions_dim)
                    + cfg.algo.world_model.recurrent_model.recurrent_state_size
                    + cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size
                ),
                output_dim=cfg.algo.world_model.stochastic_size * cfg.algo.world_model.discrete_size,
                hidden_sizes=[1024] * 5,
                activation=hydra.utils.get_class("torch.nn.SiLU"),
                flatten_dim=None,
                layer_args={"bias": ensembles_ln_cls == nn.Identity},
                norm_layer=ensembles_ln_cls,
                norm_args={
                    "eps": 0.001,
                    "normalized_shape": 1024,
                },
            ).apply(init_weights)
        )

ensembles = nn.ModuleList(ens_list)
for i in range(len(ensembles)):
    ensembles[i] = fabric.setup_module(ensembles[i])

if cfg.checkpoint.resume_from:
    state = fabric.load(cfg.checkpoint.resume_from)

world_size = fabric.world_size
buffer_size = cfg.buffer.size // int(cfg.env.num_envs * world_size) if not cfg.dry_run else 4
rb = EnvIndependentReplayBuffer(
    buffer_size,
    n_envs=cfg.env.num_envs,
    memmap=cfg.buffer.memmap,
    memmap_dir=os.path.join(".", "memmap_buffer", f"rank_{fabric.global_rank}"),
    buffer_cls=SequentialReplayBuffer,
)

if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
    if isinstance(state["rb"], list) and world_size == len(state["rb"]):
        rb = state["rb"][fabric.global_rank]
    elif isinstance(state["rb"], EnvIndependentReplayBuffer):
        rb = state["rb"]
    else:
        raise RuntimeError(f"Given {len(state['rb'])}, but {world_size} processes are instantiated")
    
world_model, actor, critic, critic_target, player = build_agent(
    fabric,
    actions_dim,
    is_continuous,
    cfg,
    observation_space,
    model_state["world_model"],
    model_state["actor"],
    model_state["critic"],
    model_state["target_critic"]
)
from tqdm import tqdm
policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1

for iter_num in tqdm(range(0, total_iters)):

    local_data = rb.sample_tensors(
        cfg.algo.per_rank_batch_size,
        sequence_length=cfg.algo.per_rank_sequence_length,
        n_samples=64,
        dtype=None,
        device=fabric.device,
        from_numpy=cfg.buffer.from_numpy,
    )
    batch = {k: v[i].float() for k, v in local_data.items()}
    data = batch

    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device
    data = {k: data[k] for k in data.keys()}
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
    posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
    posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)

    # embedded observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
            posterior, recurrent_state, batch_actions[i : i + 1], embedded_obs[i : i + 1], data["is_first"][i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        priors_logits[i] = prior_logits
        posteriors[i] = posterior
        posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    ensemble_optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer, params=ensembles.parameters(), _convert_="all"
    )

    loss = 0.0
    ensemble_optimizer.zero_grad(set_to_none=True)
    for ens in ensembles:
        out = ens(
            torch.cat(
                (
                    posteriors.view(*posteriors.shape[:-2], -1).detach(),
                    recurrent_states.detach(),
                    data["actions"].detach(),
                ),
                -1,
            )
        )[:-1]
        next_state_embedding_dist = MSEDistribution(out, 1)
        loss -= next_state_embedding_dist.log_prob(posteriors.view(sequence_length, batch_size, -1).detach()[1:]).mean()
    loss.backward()

    #print(loss.item())
    if iter_num % 1000 == 0:
        print(loss.item())

    ensemble_grad = None
    clip_gradients = 100
    if clip_gradients is not None and clip_gradients > 0:
        ensemble_grad = fabric.clip_gradients(
            module=ens,
            optimizer=ensemble_optimizer,
            max_norm=clip_gradients,
            error_if_nonfinite=False,
        )
    ensemble_optimizer.step()

    if (iter_num+1) % 5000 == 0:
        state = {
            "ensembles": ensembles.state_dict(),
            "ensemble_optimizer": ensemble_optimizer.state_dict(),
            "iter_num": iter_num * world_size,
            "batch_size": cfg.algo.per_rank_batch_size * world_size,
        }
        #\home\kkspyrid\sheeprl\checkpoints_ensembles
        ckpt_path = f"/home/kkspyrid/sheeprl/checkpoints_ensembles/ckpt_{iter_num}_{fabric.global_rank}.ckpt"
        fabric.save(ckpt_path, state)