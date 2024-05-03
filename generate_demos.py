import pathlib
import datetime
import uuid
import io
import argparse

import numpy as np 
import torch 

import utils
from human_policy import ReachPolicy, LiftPolicy, BaseHumanPolicy, PickPlacePolicy

np.set_printoptions(precision=4, suppress=True)

BOUND_MIN = np.array([-0.3, -0.4, 0.8])     
BOUND_MAX = np.array([0.3, 0.4, 1.2])


def eplen(episode):
    return len(episode['action'])

def sample_episodes(env, policy, directory, num_episodes=1, policy_obs_keys=None, render=False):
    # Save all observation keys from environment
    episodes_saved = 0
    while episodes_saved < num_episodes:
        obs = env.reset()

        if isinstance(policy, BaseHumanPolicy):
            policy.reset()

        obs_keys = list(obs.keys())
        done = False
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []

        while not done:       
            if policy_obs_keys is not None:  
                policy_obs = np.concatenate([obs[k] for k in policy_obs_keys])
            else:
                policy_obs = obs
            action, _ = policy.predict(policy_obs)
            obs, rew, done, info = env.step(action)

            if render:
                env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

        print(f"Episode return: {np.sum(episode['reward']):.2f}")
        save_episode(directory, episode)
        episodes_saved += 1
    env.close()


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def save_osc_episodes(num_episodes=64, render=False):
    """
    Generate random transitions with bound checking
    """

    env_name = "Reach"
    # env_name = "Lift"
    # env_name = "PickPlaceBread"
    controller_type = "OSC_POSE"
    # controller_type = "OSC_POSITION"
    # robots = "Panda"
    # robots = "Sawyer"
    robots = "xArm6"

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 200, "table_full_size": (0.6, 0.6, 0.00)}
    elif env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True}
        if robots == "Panda":
            env_kwargs["gripper_types"] = 'PandaTouchGripper'
        elif robots == "xArm6":
            env_kwargs["gripper_types"] = 'Robotiq85TouchGripper'
    elif env_name == "PickPlaceBread":
        policy_cls = PickPlacePolicy
        env_kwargs = {"horizon": 500, "use_touch_obs": True}

    env = utils.make_robosuite_env(env_name, robots, controller_type, render=render, **env_kwargs)
    policy = policy_cls(env)

    directory = pathlib.Path(f"./human_demonstrations/{env_name}/{robots}/{controller_type}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    episodes_saved = 0
    while episodes_saved < num_episodes:

        # episode = collect_random_episode(env, render=render)
        episode = collect_human_episode(env, policy, render=render)

        if episode is None:
            continue
        else:
            print(f"Episode return: {np.sum(episode['reward']):.2f}")
            save_episode(directory, episode)
            episodes_saved += 1

def collect_human_episode(env, policy, render=False, return_joints=False, reset_target=False):

    if reset_target:
        obs = env.reset_target()
    else:
        obs = env.reset()
    policy.reset()

    assert (obs['target_pos'] >= BOUND_MIN).all() and (obs['target_pos'] <= BOUND_MAX).all()

    # Record the mujoco states so that we load to joint env
    if return_joints:
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())   

    desired_jps, gripper_actions = [], []

    obs_keys = list(obs.keys())
    done = False
    episode = {}
    for k in obs_keys:
        episode[k] = [obs[k]]
    episode['action'] = []
    episode['reward'] = []

    while not done:       
        action, action_info = policy.predict(obs)
        obs, rew, done, info = env.step(action)

        if (obs['robot0_eef_pos'] < BOUND_MIN).any() or (obs['robot0_eef_pos'] > BOUND_MAX).any():
            print(f"Human demo out of bounds at {obs['robot0_eef_pos']}")
            if return_joints:
                return None, None
            else:
                return  

        if render:
            env.render()

        desired_jps.append(env.robots[0]._joint_positions)
        gripper_actions.append(action[-1])

        for k in obs_keys:
            episode[k].append(obs[k])
        episode['action'].append(action)
        episode['reward'].append(rew)

    print(f"Episode return: {np.sum(episode['reward']):.2f}")
    if env._check_success():
        if return_joints:
            return episode, {'task_info': [task_xml, task_init_state], 
                'joint_info': [desired_jps, gripper_actions]}
        else:
            return episode
    else:
        if return_joints:
            return None, None
        else:
            return None


def collect_random_episode(env, render=False, return_joints=False):

    obs = env.reset()

    # Record the mujoco states so that we load to joint env
    if return_joints:
        task_xml = env.sim.model.get_xml()
        task_init_state = np.array(env.sim.get_state().flatten())   

    desired_jps, gripper_actions = [], []

    obs_keys = list(obs.keys())
    done = False
    episode = {}
    for k in obs_keys:
        episode[k] = [obs[k]]
    episode['action'] = []
    episode['reward'] = []

    while not done:       
        action = np.random.uniform(low=-1, high=1, size=env.action_dim)
        action[0] += 0.04
        action = np.clip(action, -1, 1)
        obs, rew, done, info = env.step(action)

        if (obs['robot0_eef_pos'] < bound_min).any() or (obs['robot0_eef_pos'] > bound_max).any():
            if return_joints:
                return None, None
            else:
                return  

        if render:
            env.render()

        desired_jps.append(env.robots[0]._joint_positions)
        gripper_actions.append(action[-1])

        for k in obs_keys:
            episode[k].append(obs[k])
        episode['action'].append(action)
        episode['reward'].append(rew)

    if return_joints:
        return episode, {'task_info': [task_xml, task_init_state], 
            'joint_info': [desired_jps, gripper_actions]}
    else:
        return episode

def osc_to_jv(env_name, robots, num_episodes=64, render=False):

    controller_type = "OSC_POSE"
    # controller_type = "OSC_POSITION"

    if env_name == "Reach":
        policy_cls = ReachPolicy
        env_kwargs = {"horizon": 100, "table_full_size": (0.6, 0.6, 0.05)}
    elif env_name == "Lift":
        policy_cls = LiftPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}
        # env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, np.random.uniform(0.85, 0.95))}
    elif env_name == "PickPlaceBread":
        policy_cls = PickPlacePolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, 
            "bin1_pos": (-0.05, -0.25, 0.90), "bin2_pos": (-0.05, 0.28, 0.90)}
    elif env_name == "Stack":
        policy_cls = StackPolicy
        env_kwargs = {"horizon": 200, "use_touch_obs": True, "table_offset": (0, 0, 0.908)}

    if robots == "Panda":
        env_kwargs["gripper_types"] = 'PandaTouchGripper'
    elif robots == "xArm6":
        env_kwargs["gripper_types"] = 'Robotiq85TouchGripper'
    elif robots == "Sawyer":
        env_kwargs["gripper_types"] = 'RethinkTouchGripper'

    env = utils.make_robosuite_env(env_name, robots, controller_type,
        render=render, **env_kwargs)
    policy = policy_cls(env)

    jv_env = utils.make_robosuite_env(
        env_name, robots=robots, controller_type="JOINT_VELOCITY", 
        render=render, **env_kwargs)

    directory = pathlib.Path(f"./human_demonstrations/{env_name}/{robots}/JOINT_VELOCITY")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    episodes_saved = 0
    while episodes_saved < num_episodes:

        if episodes_saved % 5 == 0:
            reset_target = False
        else:
            reset_target = True
        # reset_target = False

        episode = None
        while episode is None:
            episode, ep_info = collect_human_episode(env, policy, render=render, 
                return_joints=True, reset_target=reset_target)

            # if episode is None: 
            #     reset_target = False

            # episode, ep_info = collect_random_episode(env, policy, return_joints=True)

        task_xml, task_init_state = ep_info['task_info']
        desired_jps, gripper_actions = ep_info['joint_info']

        # Reset environment to the same initial state
        jv_env.reset()
        jv_env.reset_from_xml_string(task_xml)
        jv_env.sim.reset()
        jv_env.sim.set_state_from_flattened(task_init_state)
        jv_env.sim.forward()   

        obs = jv_env._get_observations(force_update=True)
        obs_keys = list(obs.keys())
        episode = {}
        for k in obs_keys:
            episode[k] = [obs[k]]
        episode['action'] = []
        episode['reward'] = []    

        # for i, (next_jp, gripper_action, action_info) in enumerate(
        #     zip(desired_jps, gripper_actions, action_infos)):

        for i, (next_jp, gripper_action) in enumerate(
            zip(desired_jps, gripper_actions)):

            action = np.zeros(jv_env.robots[0].dof)
            action[-1] = gripper_action
            err = next_jp - jv_env.robots[0]._joint_positions

            kp = 20
            action[:-1] = np.clip(err*kp, -1, 1)

            obs, rew, done, info = jv_env.step(action)

            if (obs['robot0_eef_pos'] < BOUND_MIN).any() or (obs['robot0_eef_pos'] > BOUND_MAX).any():
                print(f"Joint vel episode out of bounds at {obs['robot0_eef_pos']}")
                break

            if render:
                jv_env.render()

            for k in obs_keys:
                episode[k].append(obs[k])
            episode['action'].append(action)
            episode['reward'].append(rew)

            if done: break

        print(f"Episode {episodes_saved} return: {np.sum(episode['reward']):.2f}")
        # print(episode['target_to_robot0_eef_pos'][-1])
        # save_episode(directory, episode)
        # episodes_saved += 1        

        if np.sum(episode['reward']) > 50:
            save_episode(directory, episode)
            episodes_saved += 1
        # if jv_env._check_success():
            # save_episode(directory, episode)
            # episodes_saved += 1

    env.close()
    jv_env.close()

def check_data(): 
    import replay_buffer
    buffer_1 = replay_buffer.ReplayBuffer(
        obs_shape=(3,),
        action_shape=(8,),
        capacity=1000000,
        batch_size=256,
        device='cpu',
    )   
    buffer_2 = replay_buffer.ReplayBuffer(
        obs_shape=(3,),
        action_shape=(8,),
        capacity=1000000,
        batch_size=256,
        device='cpu',
    )   

    obs_keys = ["robot0_eef_pos"]
    demo_dir = pathlib.Path("human_demonstrations/Reach/Panda/JOINT_VELOCITY")
    demo_paths_1 = utils.load_episodes(demo_dir, obs_keys)
    buffer_1.add_rollouts(demo_paths_1)

    demo_dir = pathlib.Path("human_demonstrations/Reach/Sawyer/JOINT_VELOCITY")
    demo_paths_2 = utils.load_episodes(demo_dir, obs_keys)
    buffer_2.add_rollouts(demo_paths_2)

    obs_1 = buffer_1.obses[:buffer_1.idx]
    obs_2 = buffer_2.obses[:buffer_2.idx]

    print(obs_1.mean(0), obs_1.std(0), np.amin(obs_1, axis=0), np.amax(obs_1, axis=0))
    print(obs_2.mean(0), obs_2.std(0), np.amin(obs_2, axis=0), np.amax(obs_2, axis=0))


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Reach", help="Robosuite task")
    parser.add_argument("--robot", type=str, default="Panda", help="Robot to generate demonstrations")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of demonstration episodes to generate")
    args = parser.parse_args()

    osc_to_jv(args.env_name, args.robot, args.num_episodes, render=False)



