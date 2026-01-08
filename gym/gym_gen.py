import gym
import numpy as np
import pickle
import os
from stable_baselines3 import PPO
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def generate_histories(num_trajs, model, env_name):
    """
    Generate LunarLander trajectories for a given number of episodes.
    """
    if env_name == "LL":
        env = gym.make("LunarLander-v2")
    elif env_name == "CP":
        env = gym.make("CartPole-v1")
    elif env_name == "AC":
        env = gym.make("Acrobot-v1")
    else:
        print("Invalid environment")
        exit(1)
        
    trajs = []
    
    with tqdm(total=num_trajs, desc=f"Generating {num_trajs} trajectories") as pbar:
        for _ in range(num_trajs):
            # Reset environment
            state = env.reset()
            
            # Initialize buffers for current episode
            states = []
            actions = []
            next_states = []
            rewards = []
            done_list = []
            
            done = False
            while not done:
              
                action, _ = model.predict(state, deterministic=True)
                # Force action to be 0 if the legs are touching the ground: This is optimal behavior
                
                next_state, reward, done, _ = env.step(action)

                if env_name == "LL":
                  
                    if (state[6] == 1) & (state[7] == 1):
                        action = 0
                        next_state = state
                        reward = 100
                        done = True
                    
                states.append(state.copy()) 
                actions.append(action)
                next_states.append(next_state.copy())
                rewards.append(reward) 
                done_list.append(done)
                
                state = next_state
            
            # Save completed trajectory
            traj = {
                'states': np.array(states),
                'actions': np.array(actions),
                'next_states': np.array(next_states),
                'rewards': np.array(rewards),
                'done': np.array(done_list)
            }
            trajs.append(traj)
            pbar.update(1)
    
    env.close()
    return trajs

def build_data_filename(config, mode):
    """
    Builds the filename for the dataset.
    Mode is either 'train', 'test', or 'eval'.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}")
    
    filename += f'_{mode}'
    
    return filename_template.format(filename)

def generate(config):
#if __name__ == "__main__":
    # Create datasets directory if it does not exist
    os.makedirs("datasets", exist_ok=True)
    
    if config["env"] == "LL":
        env_name = "LL"
        path = "gym/Expert_policy/LunarLander-v2_PPO.zip"
    elif config["env"] == "CP":
        env_name = "CP"
        env = gym.make("CartPole-v1")
        path = "gym/Expert_policy/CartPole-v1_PPO.zip"
    elif config["env"] == "AC":
        env_name = "AC"
        env = gym.make("Acrobot-v1")
        path = "gym/Expert_policy/Acrobot-v1_PPO.zip"
    else:
        print("Invalid environment")
        exit(1)
    # Configuration dictionary
    
    # Split train/test trajectories
    
    if config["num_trajs"] < 20:
        NUM_TRAIN_TRAJECTORIES = config["num_trajs"]
        NUM_TEST_TRAJECTORIES = 20
    else:    
        NUM_TRAIN_TRAJECTORIES = int(config["num_trajs"] * 0.8)
        NUM_TEST_TRAJECTORIES = config["num_trajs"] - NUM_TRAIN_TRAJECTORIES


        
    # Load the trained PPO model
    try:
        custom_objects = {"clip_range": 1, "learning_rate": 0.0003}
        model = PPO.load(path, custom_objects=custom_objects)
    except FileNotFoundError:
        print("Error: Could not find the trained PPO model file")
        exit(1)
    # Generate filenames using `config`
    train_filepath = build_data_filename({**config}, 'train')
    test_filepath = build_data_filename({**config}, 'test')  
    
    if os.path.exists(train_filepath) and os.path.exists(test_filepath):
        print(f"Data files already exist for the current configuration:")
        print(f"Train file: {train_filepath}")
        print(f"Test file: {test_filepath}")
        print("Skipping data generation.")
        return
    
    # Generate train/test trajectories
    print(f"Generating {NUM_TRAIN_TRAJECTORIES} training trajectories...")
    train_trajs = generate_histories(NUM_TRAIN_TRAJECTORIES, model, env_name)

    print(f"Generating {NUM_TEST_TRAJECTORIES} testing trajectories...")
    test_trajs = generate_histories(NUM_TEST_TRAJECTORIES, model, env_name)



    # Save the trajectories
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)

    print(f"Saved training data to {train_filepath}.")
    print(f"Saved testing data to {test_filepath}.")