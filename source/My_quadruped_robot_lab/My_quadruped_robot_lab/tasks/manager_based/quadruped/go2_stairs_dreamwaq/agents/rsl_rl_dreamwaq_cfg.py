"""Training configuration consumed by the project DreamWaQ runner."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2StairsDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "go2_stairs_dreamwaq"
    empirical_normalization = False
    clip_actions = 100.0
    history_dim = 225
    latent_dim = 16
    explicit_dim = 3
    vae_learning_rate = 1.0e-3
    vae_kl_weight = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
