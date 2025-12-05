import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-v0-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_23dof_mocap_env_cfg:Unitree23DofMocapEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_23dof_mocap_env_cfg:Unitree23DofMocapPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)