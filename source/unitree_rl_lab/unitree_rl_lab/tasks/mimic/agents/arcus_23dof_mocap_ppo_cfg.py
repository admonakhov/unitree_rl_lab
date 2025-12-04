# Copyright (c) 2024, RoboVerse community
#
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO agent configuration for Arcus 23DOF MoCap tracking.

This configuration uses asymmetric actor-critic architecture where:
- Actor (policy) sees limited observations (proprioception + motion commands)
- Critic sees privileged observations (full state including body poses)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

##
# PPO Configuration
##

@configclass
class Arcus23DofMocapPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO agent for Arcus 23DOF motion tracking."""

    seed = 42

    ##
    # Runner parameters
    ##

    runner_class_name = "OnPolicyRunner"

    # Number of environments to use
    # Adjust based on your GPU memory
    num_steps_per_env = 24  # Steps collected per environment before update
    max_iterations = 10000  # Total training iterations

    # Logging
    save_interval = 100  # Save checkpoint every N iterations
    experiment_name = "arcus_23dof_mocap"
    run_name = ""  # Auto-generated if empty

    # Logging settings
    logger = "tensorboard"  # or "wandb"
    neptune_project = "robot-learning"
    wandb_project = "isaaclab"

    # Load from checkpoint
    resume = False
    load_run = -1  # -1 for latest, or specify run name
    load_checkpoint = -1  # -1 for latest checkpoint

    ##
    # Policy (Actor) Network
    ##

    @configclass
    class PolicyCfg:
        """Actor network configuration."""

        # Network architecture
        class_name = "ActorCritic"

        # Initialization
        init_noise_std = 1.0

        # Actor (policy) network
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # or "relu", "tanh"

        # Use privileged observations for critic
        # This enables asymmetric actor-critic
        use_critic_obs = True  # Use "critic" observation group

    policy = PolicyCfg()

    ##
    # PPO Algorithm
    ##

    @configclass
    class AlgorithmCfg:
        """PPO algorithm hyperparameters."""

        class_name = "PPO"

        # PPO parameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4

        # Learning rates
        learning_rate = 1.0e-3
        schedule = "adaptive"  # "adaptive", "linear", or "constant"
        gamma = 0.99
        lam = 0.95

        # Desired KL divergence for adaptive learning rate
        desired_kl = 0.01
        max_grad_norm = 1.0

    algorithm = AlgorithmCfg()

    ##
    # Empirical normalization
    ##

    @configclass
    class EmpiricalNormalizationCfg:
        """Configuration for empirical normalization of observations and values."""

        class_name = "EmpiricalNormalization"

    empirical_normalization = EmpiricalNormalizationCfg()


# Alternative: Smaller network for faster training
@configclass
class Arcus23DofMocapPPORunnerCfgSmall(Arcus23DofMocapPPORunnerCfg):
    """Smaller PPO configuration for faster training / less GPU memory."""

    max_iterations = 5000

    @configclass
    class PolicyCfg(Arcus23DofMocapPPORunnerCfg.PolicyCfg):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]

    policy = PolicyCfg()


# High-performance configuration for powerful GPUs
@configclass  
class Arcus23DofMocapPPORunnerCfgLarge(Arcus23DofMocapPPORunnerCfg):
    """Large PPO configuration for high-end GPUs."""

    num_steps_per_env = 48
    max_iterations = 15000

    @configclass
    class PolicyCfg(Arcus23DofMocapPPORunnerCfg.PolicyCfg):
        actor_hidden_dims = [1024, 512, 256]
        critic_hidden_dims = [1024, 512, 256]
        init_noise_std = 0.8

    policy = PolicyCfg()

    @configclass
    class AlgorithmCfg(Arcus23DofMocapPPORunnerCfg.AlgorithmCfg):
        num_learning_epochs = 8
        num_mini_batches = 8
        learning_rate = 5.0e-4

    algorithm = AlgorithmCfg()
