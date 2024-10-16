import io
import gym
from gym import spaces
import numpy as np
import v8_ising_model as ising
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env


# Parameters
N_RUNS = 1  # Number of independent trajectories.
CONTROL_INTERVALS = 100  # The control network will make decisions at these intervals.
K_SIGMA = 1e-4  # Weight of sigma in the order parameter.

MIN_T = 1e-3  # Minimum allowed temperature.
MAX_T = 15
MIN_h = -4
MAX_h = 4
N=32 # lattice size
J_short = 1  # Short-range coupling constant

Ti = 0.65
Tf = 0.65
hi = -1
hf = 1

num_envs = 31  # Adjust as needed

# Used for the architecture
INIT_MODEL_PATH = "./init_policy.pth"

def make_env():
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = IsingControlEnv()
        return env
    return _init


class IsingControlEnv(gym.Env):
    def __init__(self, n_trajectories=N_RUNS):
        super(IsingControlEnv, self).__init__()
        
        self.current_magnetization = None
        self.time = None
        self.lattice = None
        self.trajectory_counter = 0
        self.rewards_list = []

        self.policy_model = torch.load(INIT_MODEL_PATH)
        print(f"Loading from {INIT_MODEL_PATH}")
        for param in self.policy_model.parameters():
            param.requires_grad = False

        # Number of trajectories to average over
        self.n_trajectories = n_trajectories
        
        self.action_space = spaces.Box(low=np.array([MIN_T, MIN_h]), high=np.array([MAX_T, MAX_h]), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64)

    def set_policy_state_dict(self, policy_state_dict):
        self.policy_state_dict = policy_state_dict
        self.policy_model.load_state_dict(self.policy_state_dict)

    def check_action_validity(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"The action {action} is out of bounds. Valid range for action is {self.action_space.low} to {self.action_space.high}.")

    def calc_shear_endpoints(self):
        # Compute endpoints using PPO (are this the correct endpoints if I'm not getting a bit flip)
        initial = np.array([0.0, -1.0])
        final = np.array([1.0, 1.0])

        # TODO: might need to reshape this depending on PPO format 
        action_initial, _ = self.policy_model.predict(initial, deterministic=True)
        action_final, _ = self.policy_model.predict(final, deterministic=True)

        self.T_initial, self.h_initial = action_initial
        self.T_final, self.h_final = action_final

        
    def reset(self):
        # set time and magentization to initial values
        self.current_magnetization = -1.0
        self.time = 0.0
        # reinitalize lattice
        self.lattice = ising.initialize_lattice(N)
        # zero out entropy counters
        self.initial_entropy = 0.0
        self.final_entropy = 0.0 
        self.step_entropy = 0.0 
        entropy = 0.0

        # Compute endpoints using PPO (are this the correct endpoints if I'm not getting a bit flip)
        initial = np.array([0.0, -1.0])
        final = np.array([1.0, 1.0])

        # TODO: might need to reshape this depending on PPO format
      
        action_initial, _ = self.policy_model.predict(initial, deterministic=True)
        action_final, _ = self.policy_model.predict(final, deterministic=True)


        self.T_initial, self.h_initial = action_initial
        self.T_final, self.h_final = action_final

        
        return np.array([self.time, self.current_magnetization])

    def step(self, action):
        self.calc_shear_endpoints()
        self.check_action_validity(action)
        T, h = action

        if self.time == 0:
            T = self.T_initial
            h = self.h_initial
        if self.time >= 1.0:
            T = self.T_final
            h = self.h_final

        T = T + (1-self.time)*(Ti - self.T_initial) + self.time*(Tf-self.T_final)
        # force T to be no less than 1 e-3
        if T < 1e-3:
            T = 1e-3
        h = h + (1-self.time)*(hi - self.h_initial) + self.time*(hf-self.h_final)

        beta = 1.0 / T


        self.lattice, energies, magnetizations, entprod, delta_eng = ising.simulate_ising_model(N, J_short, h, beta, 1, self.lattice)

        # add values to counters
        # add stepwise entropy to counter term
        self.step_entropy += entprod

        if self.time == 0:
            self.initial_entropy = beta*energies

        # Update the current_magnetization and time for the next step
        self.current_magnetization = magnetizations

        # Set standard reward and state
        reward = -(abs(self.current_magnetization - 1) + K_SIGMA*(-1*self.step_entropy))
        done = False

        # Store final state and inputs
        final_state = np.array([self.time, self.current_magnetization])
        final_inputs = np.array([T, h])

        # Compute the reward if trajectory is finished
        if self.time >= 1.0:
            #print(f'T, h at t = 1, {T, h}')
            # calculate final entropy
            self.final_entropy = beta*energies
            reward = float(-(abs(self.current_magnetization - 1) + K_SIGMA * (self.final_entropy - self.initial_entropy - self.step_entropy)))

            done = True
            self.rewards_list.append(reward)
            self.trajectory_counter += 1

            # Average trajectories if we have finished all desired trajectories
            if self.trajectory_counter >= self.n_trajectories:
                reward = np.mean(self.rewards_list)
                # zero out rewards list and trajectory counter
                self.rewards_list = []
                self.trajectory_counter = 0

            # Calculate entropy to return
            entropy = self.final_entropy - self.initial_entropy - self.step_entropy
            print(f'Entropy: {entropy}, final-initial {self.final_entropy - self.initial_entropy}, step_ent {-self.step_entropy}')

            # Reset
            self.reset()
            

        # If not done, then set entropy to None
        else:
            self.time += 1.0 / CONTROL_INTERVALS
            entropy = None

        return final_state, reward, done, {'inputs':final_inputs, 'entropy': entropy}
    
    def render(self, mode='human'):
        # TODO: Fix visualizations and rendering
        pass
    
    def close(self):
        pass


class UpdateModelCallback(BaseCallback):
    '''
    This class is a callback to update the model policy used to calculate the endpoints for the shear
    '''
    def __init__(self, env, update_interval):
        super(UpdateModelCallback, self).__init__()
        self.env = env
        self.update_interval = update_interval
        self.step_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1

        # Check if the step counter has reached the update interval
        if self.step_counter % self.update_interval == 0:
            self.env.env_method("set_policy_state_dict", self.model.policy.state_dict())

        return True

class CustomMetricsCallback(BaseCallback):
    """
    A custom callback that logs additional values to TensorBoard, leaving here in case I need it later
    """
    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Example: logging gradient norms
        # Note: This is a placeholder and may need to be adapted based on your algorithm/network structure
        # gradient_norms = np.linalg.norm(self.model.policy.parameters(), 2).item()
        # self.logger.record('train/gradient_norm', gradient_norms)

        # Log other custom metrics as necessary
        # ...
        return True

def main():
    # Create vectorized environments
    envs = [make_env() for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    
    print('envs created')
    # initalize update PPO policy in environment at the end of each trajectory 
    policy_update_callback = UpdateModelCallback(env, update_interval=CONTROL_INTERVALS)

    # Set up TensorBoard callback
    tensorboard_log_dir = "./tensorboard_logs/"
    
    # Combine all callbacks
    all_callbacks = [policy_update_callback, CustomMetricsCallback()]

    # Create and train the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
    env.env_method("set_policy_state_dict", model.policy.state_dict())

    # Train PPO agent
    model.learn(total_timesteps=5000000, callback=all_callbacks)

    # Save the model
    model.save("ppo_ising_control_v10")

    # Test the trained agent
    obs = env.reset()
    time_input_list = []
    m_input_list = []
    temp_input_list = []
    h_input_list = []
    reward_list = []
    entropy_list = []
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        for j in range(num_envs):
            time_input_list.append(obs[j][0])
            m_input_list.append(obs[j][1])
            temp_input_list.append(info[j]['inputs'][0])
            h_input_list.append(info[j]['inputs'][1])
            reward_list.append(rewards[j])
            entropy_list.append(info[j]['entropy'])

    print(f'Entropy list {entropy_list}')
    print(f'Reward list {reward_list}') 

    plt.scatter(time_input_list, m_input_list)
    plt.xlabel('Time')
    plt.ylabel('Internal Magnetic Field')
    plt.title('Time vs Internal Magnetic Field')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(-1, 1)

    plt.savefig("time_vs_int_magnetic_PPO_new_shear_v10.png")
    print(f"Plot saved as time_vs_int_magnetic_PPO.png")

    plt.clf()
    plt.figure()
    plt.scatter(temp_input_list, h_input_list)
    plt.xlabel('Temperature')
    plt.ylabel('External Magnetic Field')
    plt.title('Temperature vs External Magnetic Field')
    plt.grid(True)
    plt.xlim(-1, 15)
    plt.ylim(-4, 4)

    plt.savefig("temp_vs_ext_magnetic_PPO_new_shear_v10.png")
    print(f"Plot saved as temp_vs_ext_magnetic_PPO.png")




if __name__ == '__main__':
    main()

