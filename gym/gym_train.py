import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
from mlp import MLP
from datetime import datetime



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENV_DATASET_FILES = {
    "LL": "D_LunarLander_medium_mixed.npz",
    "CP": "D_Cartpole_medium_mixed.npz",
    "AC": "D_Acrobot_medium_mixed.npz",
}
ENV_ALIASES = {
    "LL": ["lunarlander"],
    "CP": ["cartpole"],
    "AC": ["acrobot"],
}


def resolve_npz_path(config):
    dataset_file = config.get("dataset_file")
    if dataset_file and os.path.isfile(dataset_file):
        return dataset_file

    gym_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(gym_dir, ".."))
    dataset_dir_config = config.get("dataset_dir", "datasets")
    if os.path.isabs(dataset_dir_config):
        dataset_dirs = [dataset_dir_config]
    else:
        dataset_dirs = [
            os.path.join(gym_dir, dataset_dir_config),
            os.path.join(repo_root, dataset_dir_config),
        ]

    env = config.get("env")
    filename = ENV_DATASET_FILES.get(env)
    aliases = ENV_ALIASES.get(env, [])

    for dataset_dir in dataset_dirs:
        if not os.path.isdir(dataset_dir):
            continue
        if filename:
            candidate = os.path.join(dataset_dir, filename)
            if os.path.isfile(candidate):
                return candidate
        candidates = [f for f in os.listdir(dataset_dir) if f.endswith(".npz")]
        for alias in aliases:
            for candidate in candidates:
                if alias in candidate.lower():
                    return os.path.join(dataset_dir, candidate)
    return None

class Dataset(torch.utils.data.Dataset):
    """Optimized Dataset class for storing and sampling (s, a, r, s') transitions."""

    def __init__(self, path, config):
        self.store_gpu = config.get('store_gpu', False)  # Store tensors on GPU if True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure path is a list
        if not str(path).endswith(".npz"):
            raise ValueError(f"Expected an .npz dataset file, got: {path}")

        # Load dataset efficiently
        data = np.load(path)
        states = np.asarray(data["obs"])  # dimension is transitions x state_dim
        actions = np.asarray(data["act"])
        next_states = np.asarray(data["obs2"])
        rewards = np.asarray(data["rew"])
        done = np.asarray(data["done"])
        self.episode_returns = np.asarray(data.get("episode_returns", []))

        
        # Convert to PyTorch tensors
        self.dataset = {
            'states': self.convert_to_tensor(states, store_gpu=self.store_gpu),
            'actions': self.convert_to_tensor(actions, store_gpu=self.store_gpu),
            'next_states': self.convert_to_tensor(next_states, store_gpu=self.store_gpu),
            'rewards': self.convert_to_tensor(rewards, store_gpu=self.store_gpu),
            'done': self.convert_to_tensor(done, store_gpu=self.store_gpu)
        }

        # Shuffle dataset at initialization 
        if config.get('shuffle', False):
            self.shuffle_dataset()

    def __len__(self):
        """Return the number of transitions."""
        return len(self.dataset['states'])

    def __getitem__(self, idx):
        """Return a single (s, a, r, s') transition."""
        return {
            'states': self.dataset['states'][idx],
            'actions': self.dataset['actions'][idx],
            'next_states': self.dataset['next_states'][idx],
            'rewards': self.dataset['rewards'][idx],
            'done': self.dataset['done'][idx]
        }

    def shuffle_dataset(self):
        """Shuffle all transitions."""
        indices = np.arange(len(self.dataset['states']))
        np.random.shuffle(indices)
        
        for key in self.dataset.keys():
            self.dataset[key] = self.dataset[key][indices]
        
    
    @staticmethod
    def convert_to_tensor(x, store_gpu):
        """Convert numpy array to tensor, optionally storing on GPU."""
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        return tensor.to("cuda") if store_gpu else tensor



def build_data_filename(config, mode):
    """
    Builds the filename for the data.
    Mode is either 'train', 'test', or 'eval'.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}")
    
    filename += f'_{mode}'
    
    return filename_template.format(filename)


def build_model_filename(config):
    """
    Builds the filename for the model.
    """
    filename = (f"{config['env']}_shuf{config['shuffle']}_lr{config['lr']}"
                f"_decay{config['decay']}_Tik{config['Tik']}"
                f"_do{config['dropout']}_embd{config['n_embd']}"
                f"_layer{config['n_layer']}_head{config['n_head']}"
                f"_seed{config['seed']}")
    return filename

def build_log_filename(config):
    """
    Builds the filename for the log file.
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_lr{config['lr']}"
                f"_batch{config['batch_size']}"
                f"_decay{config['decay']}"
                f"_clip{config['clip']}"
                )
    filename += f'_{timestamp}'
    
    return filename + ".log"

def printw(message, config):
    print(message)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = build_log_filename(config)
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, "a") as log_file:
        print(message, file=log_file)
        
def corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    covariance = torch.mean((x - x_mean) * (y - y_mean))  # Cov(X, Y)
    std_x = torch.std(x, unbiased=False)  # Standard deviation of X
    std_y = torch.std(y, unbiased=False)  # Standard deviation of Y

    correlation = covariance / (std_x * std_y)  # Pearson correlation formula
    return correlation        

def train(config):
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    if not os.path.exists('logs'):
        os.makedirs('logs', exist_ok=True)

    # Set random seeds
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Prepare dataset
    dataset_config = {
        'num_trajs': config['num_trajs'],
        'store_gpu': True,
        'shuffle': config['shuffle'],
        'env': config['env']
    }

    npz_path = resolve_npz_path(config)
    if not npz_path:
        raise FileNotFoundError(
            "Offline dataset not found. Expected dataset_dir to contain "
            f"{ENV_DATASET_FILES.get(config.get('env'))} or pass dataset_file."
        )
    dataset_config['dataset_file'] = npz_path
    train_dataset = Dataset(npz_path, dataset_config)
    test_dataset = train_dataset


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )

    if config['env'] == 'LL':
        states_dim = 8
        actions_dim = 4
        init_b_value = 100
    elif config['env'] == 'CP':
        states_dim = 4
        actions_dim = 2
        init_b_value = 1
    elif config['env'] == 'AC':
        states_dim = 6
        actions_dim = 3
        init_b_value = -10
    else:
        print('Invalid environment')
        exit()
        
    def custom_output_b_init(bias):
        nn.init.constant_(bias, init_b_value)
    
    # Prepare model
    model_config = {
        'hidden_sizes' : [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], #layer normalization
    }
    model = MLP(states_dim, actions_dim, output_b_init=custom_output_b_init, **model_config).to(device)
    
    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    MSE_loss_fn = torch.nn.MSELoss(reduction='mean')
    MAE_loss_fn = torch.nn.L1Loss(reduction='mean')
    
    repetitions = config['repetitions']  # Number of repetitions

    episode_returns = np.asarray(getattr(train_dataset, "episode_returns", []))
    if episode_returns.size > 0:
        printw(
            "Episode returns summary - "
            f"count={episode_returns.size}, "
            f"mean={episode_returns.mean():.4f}, "
            f"std={episode_returns.std():.4f}, "
            f"min={episode_returns.min():.4f}, "
            f"max={episode_returns.max():.4f}",
            config,
        )
        
    rep_test_r_MAPE_loss = []
    rep_best_r_MAPE_loss = []
    rep_episode_returns = []

    
    for rep in range(repetitions):
        print(f"\nStarting repetition {rep+1}/{repetitions}")
        if episode_returns.size > 0:
            rep_episode_returns.append(episode_returns)
            
        train_loss = []
        train_be_loss = []
        train_ce_loss = []
        train_D_loss = []
        test_r_MAPE_loss = []
        
        #Storing the best training epoch and its corresponding best Q MSE loss/Q values
        best_epoch = -1
        best_r_MAPE_loss = 9999
      
        
        for epoch in tqdm(range(config['num_epochs']), desc="Training Progress"):
            
            ############### Start of an epoch ##############
            
            ### EVALUATION ###
            printw(f"Epoch: {epoch + 1}", config)
            start_time = time.time()
            with torch.no_grad():
                epoch_r_MAPE_loss = 0.0
                
                ##### Test batch loop #####
                
                for i, batch in enumerate(test_loader):
                    print(f"Batch {i} of {len(test_loader)}", end='\r')
                    batch = {k: v.to(device) for k, v in batch.items()} 
                    states = batch['states']
                    pred_q_values, pred_q_values_next, pred_vnext_values = model(batch) #dimension is (batch, action_dim)
                    
                    true_actions = batch['actions'].long() #dimension is batch
        
                    pred_r_values = pred_q_values - config['beta']*pred_vnext_values 
                    chosen_pred_r_values = torch.gather(pred_r_values, dim=1, index=true_actions.unsqueeze(-1)) 

                    true_r_values = batch['rewards'] #dimension
              
                    
                    ##############For computing r_MAPE (mean absolute percentage error)########################
                    diff = torch.abs(chosen_pred_r_values - true_r_values) #dimension is (batch_size, horizon)
                    #denom = torch.abs(true_r_values) #dimension is (batch_size, horizon)
                    #r_MAPE = torch.mean(diff / denom)*100 #dimension is (1,), because it is the mean of all diff/denom values in the batch
                    r_MAPE = torch.mean(diff)
                    epoch_r_MAPE_loss += r_MAPE.item()
                    ##############Computing correlation coefficient########################
                    #pearson_corr = corr(chosen_pred_r_values.squeeze(), true_r_values.squeeze())
                    #epoch_r_MAPE_loss += pearson_corr.item()
                    
                    
                
                ##### Finish of the batch loop for a single epoch #####
                ##### Back to epoch level #####
                # Note that epoch MSE losses are sum of all test batch means in the epoch
                
                if epoch_r_MAPE_loss/len(test_loader) < best_r_MAPE_loss: #epoch_r_MAPE_loss is sum of all test batch means in the epoch
        
                    best_r_MAPE_loss = epoch_r_MAPE_loss/len(test_loader) #len(test_dataset) is the number of batches in the test dataset
                    best_epoch = epoch          
            
            ############# Finish of an epoch's evaluation ############
    
            test_r_MAPE_loss.append(epoch_r_MAPE_loss / len(test_loader)) #mean of all test batch means in the epoch
            
            end_time = time.time()
            #printw(f"\tCross entropy test loss:
            
            printw(f"\tMAPE of r(s,a): {test_r_MAPE_loss[-1]}", config)

            printw(f"\tEval time: {end_time - start_time}", config)
            
            
            ############# Start of an epoch's training ############
            
            epoch_train_loss = 0.0
            epoch_train_be_loss = 0.0
            epoch_train_ce_loss = 0.0
            epoch_train_D_loss = 0.0
            start_time = time.time()
            
            torch.autograd.set_detect_anomaly(True)
            
            
            for i, batch in enumerate(train_loader): #For batch i in the training dataset
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()} #dimension is (batch_size, horizon, state_dim)
                
                pred_q_values, pred_q_values_next, pred_vnext_values  = model(batch) 

                true_actions = batch['actions'].long() 
                states = batch['states']
                
                true_rewards = batch['rewards']
                
                ### Q(s,a) 
                chosen_q_values = torch.gather(pred_q_values, dim=1, index=true_actions.unsqueeze(-1)) #dimension is (batch_size, horizon)
                chosen_vnext_values = torch.gather(pred_vnext_values, dim=1, index=true_actions.unsqueeze(-1)) #dimension is (batch_size, horizon)
                
                #Empirical V(s') = logsumexp Q(s',a') + gamma
                logsumexp_nextstate = torch.logsumexp(pred_q_values_next, dim=1) #dimension is (batch_size*horizon,)
        
                vnext = logsumexp_nextstate
                done = batch['done'].to(torch.bool)
                vnext = torch.where(done, torch.tensor(0.0, device=vnext.device), vnext)
                
                
                #D update only. Fitting V(s') prediction to logsumexp Q(s',a) prediction
                if i % 2 == 0: # update xi only, update xi every 2 batches

                    #V(s')-E[V(s')] minimization loss
                    D = MSE_loss_fn(vnext.clone().detach(), chosen_vnext_values)
                    D.backward()
                    
                    #Non-fixed lr part starts
                    current_lr_vnext = config['lr'] / (1 + config['decay']*epoch)
                    vnext_optimizer.param_groups[0]['lr'] = current_lr_vnext
                    #Non-fixed lr part ends
                    
                    if config['clip'] != False:                    
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip'])
                    
                    vnext_optimizer.step() #we use separate optimizer for vnext
                    vnext_optimizer.zero_grad() #clear gradients for the batch
                    epoch_train_D_loss += D.item()  #per-sample loss
                    model.zero_grad() #clear gradients for the batch. This prevents the accumulation of gradients.
            
                else:  # update Q only, update Q every 2 batches
                    
                    # Mean_CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                    # ce_loss = Mean_CrossEntropy_loss_fn(pred_q_values, true_actions) #shape  is (batch_size*horizon,)
                  
                    #Setting pivot reward does not affect anything. So whatever we fix it it does not
                    # harm or benefit the outcome. However, for evaluation convenience, 
                    #we give the true reward for as the pivot action's reward.
            
                    #Just put true rewards for all actions in the batch for now, to calculate td
                    pivot_rewards = true_rewards 
                    
                    td_error = chosen_q_values - pivot_rewards- config['beta'] * vnext #\delta(s,a) = Q(s,a) - r(s,a) - beta*V(s')                    
                    td_error = torch.where(done, chosen_q_values - pivot_rewards, td_error)

                    
                    vnext_dev = (vnext - chosen_vnext_values.clone().detach())
                    #Bi-conjugate trick to compute the Bellman error
                    be_error_naive = td_error**2-config['beta']**2 * vnext_dev**2 #dimension is (batch_size*horizon,)
                    #We call it naive because we just add pivot r for every actions we see in the batch
                    
                    # #At terminal state, terminal action is trival. Set it as pivot action.
                    # if config['env'] == 'LL':
                    #     #For LunarLander, set the pivot action to be action 2. 
                    #     indices_TF = (true_actions == 2) | done
                    # if config['env'] == 'AC':
                    #     #For Acrobot, set the pivot action to be 0.
                    #     indices_TF = (true_actions == 0) | done
                    # if config['env'] == 'CP':
                    #     #For CartPole, set the pivot action to be 0.
                    #     indices_TF = (true_actions == 0) | done
                        
                    be_error_0 = be_error_naive
                    
                    mean_MAE_loss_fn = torch.nn.L1Loss(reduction='mean')
                    
                    be_loss = mean_MAE_loss_fn(be_error_0, torch.zeros_like(be_error_0))
                    #number of action=2 in the batch does not matter, as we normalize the loss by the number of action=2 in the batch
                    
                    #Tikhonov
                    # if config['Tik'] == True:
                    #     loss = 100*(1/(1+epoch))*ce_loss + be_loss
                    # else:               
                    #     loss = 10*ce_loss + be_loss
                    #
                    
                    be_loss.backward()
                    
                    #Non-fixed lr part starts
                    current_lr_q = config['lr'] / (1 + config['decay']*epoch)
                    q_optimizer.param_groups[0]['lr'] = current_lr_q
                    #Non-fixed lr part ends
                    
                    if config['clip'] != False:                    
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip'])
                    q_optimizer.step()
                    q_optimizer.zero_grad() #clear gradients for the batch
                
                    model.zero_grad()
                    
                    # epoch_train_loss += loss.item() 
                    epoch_train_be_loss += be_loss.item() 
                    # epoch_train_ce_loss += ce_loss.item() 
                    
                    # print(f"Epoch_train_loss: {epoch_train_loss}", end='\r')

                
                if i == 0: #i=0 means the first batch
                    pred_r_values_print = pred_q_values[:10,:] - config['beta']*pred_vnext_values[:10,:] #for print
                    chosen_r_values_print = torch.gather(pred_r_values_print, dim=1, index=true_actions[:10].unsqueeze(-1)) #for print
                    true_r_values_print = true_rewards[:10].unsqueeze(1) #for print
                    actions_print = true_actions[:10].int().unsqueeze(1) #for print
                
                    pred_r_values_with_true_r = torch.cat((actions_print, true_r_values_print, chosen_r_values_print), dim=1) #dimension is (batch_size, state_dim+action_dim)
                    pred_r_values_np = pred_r_values_with_true_r.cpu().clone().detach().numpy()
                    np.set_printoptions(suppress=True, precision=6)
                    printw(f"Predicted r values: {pred_r_values_np}", config)
                
       
                
                
            #len(train_dataset) is the number of batches in the training dataset
            # train_loss.append(epoch_train_loss / len(train_loader)) 
            train_be_loss.append(epoch_train_be_loss / len(train_loader))
            # train_ce_loss.append(epoch_train_ce_loss / len(train_loader))
            train_D_loss.append(epoch_train_D_loss / len(train_loader))

            end_time = time.time()
            
            # printw(f"\tTrain loss: {train_loss[-1]}", config)
            printw(f"\tBE loss: {train_be_loss[-1]}", config)
            # printw(f"\tCE loss: {train_ce_loss[-1]}", config)
            printw(f"\tTrain time: {end_time - start_time}", config)


            # Logging and plotting
            
            if (epoch + 1) % 1000 == 0:
                torch.save(model.state_dict(),
                    f'models/{build_log_filename(config)}_rep{rep}_epoch{epoch+1}.pt')

            if (epoch + 1) % 1 == 0:
                plt.figure(figsize=(12, 12))  # Increase the height to fit all plots
    
                # # Plotting total train loss
                # plt.subplot(5, 1, 1) # Adjust to 6x1 grid
                # plt.yscale('log')
                # plt.xlabel('epoch')
                # plt.ylabel('Total Train Loss')
                # plt.plot(train_loss[1:], label="Total Train Loss")
                # plt.legend()

                # Plotting BE loss
                plt.subplot(5, 1, 2) # Second plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Train BE Loss')
                plt.plot(train_be_loss[1:], label="Bellman Error Loss", color='red')
                plt.legend()

                # # Plotting CE loss
                # plt.subplot(5, 1, 3) # Third plot in a 6x1 grid
                # plt.yscale('log')
                # plt.xlabel('epoch')
                # plt.ylabel('Train CE Loss')
                # plt.plot(train_ce_loss[1:], label="Cross-Entropy Loss", color='blue')
                # plt.legend()
                
                # Plotting r MAPE loss 
                plt.subplot(5, 1, 4) # Fifth plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Test R MAPE Loss')
                plt.plot(test_r_MAPE_loss[1:], label="r MAPE Loss", color='purple')
                plt.legend()
                
                plt.subplot(5, 1, 5) # Sixth plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('D Loss')
                plt.plot(train_D_loss[1:], label="D Loss", color='orange')
                plt.legend()
                
                
                plt.tight_layout()
                plt.savefig(f"figs/loss/{build_log_filename(config)}_rep{rep}_losses.png")
                plt.close()
                

            ############### Finish of an epoch ##############
        ##### Finish of all epochs #####
        
        printw(f"Best epoch for repetition {rep+1} : {best_epoch}", config)
        printw(f"Best R MAPE loss for repetition {rep+1}: {best_r_MAPE_loss}", config)
        
        ################## Finish of one repetition #########################      
        if best_epoch > 0:
            rep_best_r_MAPE_loss.append(best_r_MAPE_loss) 
        else:
            printw("No best r values were recorded during training.", config)  
            
        rep_test_r_MAPE_loss.append(test_r_MAPE_loss)
            
        torch.save(model.state_dict(), f'models/{build_log_filename(config)}.pt')
        
        printw(f"\nTraining of repetition {rep+1} finished.", config)
        
    #### Finish of all repetitions ####    
    rep_test_r_MAPE_loss = np.array(rep_test_r_MAPE_loss) #dimension is (repetitions, num_epochs)
    
    mean_r_mape = np.mean(rep_test_r_MAPE_loss, axis=0) #dimension is (num_epochs,)
    std_r_mape = np.std(rep_test_r_MAPE_loss, axis=0)/np.sqrt(repetitions)
    
    epochs = np.arange(0, config['num_epochs'])
    
    plt.figure(figsize=(12, 6))  # Increase the height to fit all plots

    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('R MAPE Loss')
    plt.plot(mean_r_mape, label="Mean R MAPE Loss", color='blue')
    plt.fill_between(epochs, mean_r_mape - std_r_mape, mean_r_mape + std_r_mape, alpha=0.2, color='blue')
    plt.legend()

 
    plt.tight_layout()
    plt.savefig(f"figs/loss/Reps{repetitions}_{build_log_filename(config)}_losses.png")
    plt.close()
    
    if rep_episode_returns:
        plt.figure()
        all_returns = []
        for returns in rep_episode_returns:
            steps = np.arange(len(returns))
            all_returns.append(returns)
            plt.plot(steps, returns, alpha=0.3)
        mean_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        plt.plot(steps, mean_returns, linewidth=2, color='red')
        plt.fill_between(steps, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
        env_name = config.get("env", "env")
        plt.title(f"SBEED ({env_name} offline mixed)")
        plt.xlabel("Gradient Updates")
        plt.ylabel("Average Episode Reward")
        plt.savefig(f"figs/loss/Reps{repetitions}_{build_log_filename(config)}_episode_returns.png")
        plt.close()    
    
    
    printw(f"\nTraining completed.", config)
    mean_best_r_mape = np.mean(rep_best_r_MAPE_loss) 
    std_best_r_mape = np.std(rep_best_r_MAPE_loss)/np.sqrt(repetitions)
    ##separate logging for the final results
    printw(f"\nFinal results for {repetitions} repetitions", config)
    printw(f"Mean best R MAPE loss: {mean_best_r_mape}", config)
    printw(f"Standard error of best R MAPE loss: {std_best_r_mape}", config)

    