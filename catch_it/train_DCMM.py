from __future__ import annotations

import hydra
import torch
import os
import random
import numpy as np
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.utils.util import omegaconf_to_dict
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_hoop import PPO_Hoop
import gymnasium as gym
import gym_dcmm
import datetime
import pytz
# os.environ['MUJOCO_GL'] = 'egl'
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    config.test = config.test
    model_path = None
    if config.task != 'HoopCatch':
        raise ValueError("Invalid task: {}".format(config.task))
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)
        model_path = config.checkpoint

    # use the device for rl
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'

    # Resolve the seed once and keep it as an integer.
    seed = int(config.seed)
    if seed == -1:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    elif seed < 0:
        raise ValueError("seed must be -1 or a non-negative integer")

    config.seed = seed

    # Seed random sources in the main process.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Using seed: {seed}")

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # Create and wrap the environment
    env_name = 'gym_dcmm/DcmmVecWorld-v0'
    print("config.num_envs: ", config.num_envs)
    env = gym.make_vec(env_name, num_envs=int(config.num_envs), 
                    task=config.task, camera_name=["top"],
                    render_per_step=False, render_mode = "rgb_array",
                    object_name = "object",
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = False, print_info = False,
                    print_reward = False, print_ctrl = False,
                    print_contacts = False,
                    env_time = 2.5, steps_per_policy = 20)

    output_dif = os.path.join('outputs', config.output_name)
    # Get the local date and time
    local_tz = pytz.timezone('Asia/Shanghai')
    current_datetime = datetime.datetime.now().astimezone(local_tz)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H-%M-%S")
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)

    agent = PPO_Hoop(env, output_dif, full_config=config)

    cprint('Start Training/Testing the Agent', 'green', attrs=['bold'])
    if config.test:
        if model_path:
            print("checkpoint loaded")
            agent.restore_test(model_path)
        print("testing")
        agent.test()
    else:
        # connect to wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.output_name,
            config=omegaconf_to_dict(config),
            mode=config.wandb_mode
        )

        agent.restore_train(model_path)
        agent.train()

        # close wandb
        wandb.finish()

if __name__ == '__main__':
    main()
