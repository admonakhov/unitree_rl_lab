import gymnasium as gym

gym.register(
    id="Unitree-Arcus-A1-v0-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arcus_23dof_mocap_env_cfg:Arcus23DofMocapEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.arcus_23dof_mocap_env_cfg:Arcus23DofMocapPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.arcus_23dof_mocap_ppo_cfg:Arcus23DofMocapPPORunnerCfg",
    },
)