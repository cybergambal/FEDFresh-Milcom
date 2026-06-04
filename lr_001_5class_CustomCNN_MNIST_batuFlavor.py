import torch
import torch.optim as optim
from torch import nn
import numpy as np
import os
import time
import shutil
import argparse
from datetime import datetime
import sys
import random as rnd
from FL_setting_NeurIPS_batuFlavor import FederatedLearning
from utils import get_data_loaders, get_Model, evaluate_per_label_accuracy, save_data_to_csv

# Start time
start_time = time.time()
# Command-line arguments.
# The defaults below reproduce the experiment configuration that was previously
# hardcoded into sys.argv, so running this file with no arguments behaves
# exactly as before. The grid runner (run_grid.py) overrides only --temp,
# --user_prob_disc and --selected_mode; every other setting is held fixed.
parser = argparse.ArgumentParser(description="Federated Learning with Slotted ALOHA and CIFAR-10 Dataset", fromfile_prefix_chars='@')
parser.add_argument('--learning_rate_client', type=float, default=0.01, help='Learning rate for client training')
parser.add_argument('--learning_rate_server', type=float, default=0.1, help='Learning rate for server training')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=400, help='Batch size for training')
parser.add_argument('--num_users', type=int, default=100, help='Number of users in federated learning')
parser.add_argument('--fraction', type=float, nargs='+', default=[1.0], help='Fraction for top-k sparsification')
parser.add_argument('--num_timeframes', type=int, default=1000, help='Number of timeframes for simulation')
parser.add_argument('--seeds', type=int, nargs='+', default=[56], help='Random seeds for averaging results')
parser.add_argument('--num_runs', type=int, default=1, help='Number of simulations')
parser.add_argument('--selected_mode', type=str, default='async_asymp_EI', help='FL mode: async_asymp_EI, async_asymp_age, async_asymp_cossim, async_asymp_random, async_asymp_fresh, fedbuff, fedstale')
parser.add_argument('--cos_similarity', type=int, default=4, help='What type of cosine similarity we want to test: cos2 = 2, cos4 = 4, ...')
parser.add_argument('--train_mode', type=str, default='all', help='Which part of network we are training: all, dense, conv')
parser.add_argument('--bufferLimit', type=int, default=10, help='Buffer size limit / per-round client budget K')
parser.add_argument('--theta_inner', type=float, default=0.1, help='Theta coefficient for inner product test')
parser.add_argument('--data_mode', type=str, default='CIFAR', help='Dataset mode: MNIST or CIFAR')
parser.add_argument('--unit_gradients', type=int, default=0, help='Whether to use unit gradients 0=False, 1=True')
parser.add_argument('--adam', type=int, default=0, help='Whether to use FedAdam optimizer 0=False, 1=True')
parser.add_argument('--temp', type=float, default=0.2, help='alpha-fair parameter (alpha): 0=throughput/greedy, 1=proportional fairness, inf=equal impact')
parser.add_argument('--cos_similarity_type', type=int, default=0, help='Type of cosine similarity calculation: 0=lowest, 1=highest')
parser.add_argument('--user_prob_disc', type=float, default=0.45, help='user probability discrepancy parameter, must satisfy -0.5 < disc < 0.5')
parser.add_argument('--cuda', type=int, default=1, help='CUDA device number to use')
parser.add_argument('--out_dir', type=str, default=None, help='If set, move the results folder here after the run (overwrites an existing folder at that path)')

args = parser.parse_args()

# Parsed arguments
learning_rate_client = args.learning_rate_client
learning_rate_server = args.learning_rate_server
epochs = args.epochs
batch_size = args.batch_size
num_users = args.num_users
fraction = args.fraction
num_timeframes = args.num_timeframes
seeds_for_avg = args.seeds
num_runs = args.num_runs
selected_mode = args.selected_mode
train_mode = args.train_mode
cos_similarity = args.cos_similarity
bufferLimit = args.bufferLimit
theta_inner = args.theta_inner
data_mode = args.data_mode
unit_gradients =  False if args.unit_gradients == 0 else True
adam = False if args.adam == 0 else True
temp = args.temp
cos_similarity_type = args.cos_similarity_type
user_prob_disc = args.user_prob_disc
cuda_device = args.cuda

# Device configuration
device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
print(f"\n{'*' * 50}\n*** Using device: {device} ***\n{'*' * 50}\n")


# Initialize accuracy storage
accuracy_distributions = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

contribution_distributions = {
    run: {
        seed_index: {user: None for user in range(num_users)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

chosen_users_over_time = {
    run: {
        seed_index: {timeframe: {user: 0 for user in range(num_users)} for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

expected_gradient_magnitude = {
    run: {
        seed_index: {user: None for user in range(num_users)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

#Load model
Model = get_Model(data_mode, train_mode=train_mode)

# Main training loop
for run in range(num_runs):
    print(f"************ Run {run + 1} ************")

    for seed_index, seed in enumerate(seeds_for_avg):
        rnd.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load data
        TrainSetUsers, testloader = get_data_loaders(data_mode, batch_size, num_users)

        print(f"************ Seed {seed_index} ************")
        
        # Define number of classes based on the dataset
        num_classes = 10  # CIFAR-10 has 10 classes

        # Initialize the model
       
        model = Model(num_classes=num_classes).to(device)
        
        
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate_client, weight_decay=1e-4)

        
        keepProbAvail = np.concatenate([
            np.full(num_users // 2, 0.5 - user_prob_disc),  # First half: 0.1
            np.full(num_users - num_users // 2, 0.5 + user_prob_disc)  # Second half: 0.9
        ])
        keepProbNotAvail = np.concatenate([
            np.full(num_users // 2, 0.5 + user_prob_disc),  # First half: 0.9
            np.full(num_users - num_users // 2, 0.5 - user_prob_disc)  # Second half: 0.1
        ])

        #Initialize FL system once and for all for this seed.
        fl_system = FederatedLearning(
            selected_mode, num_users, device,
            cos_similarity, model, TrainSetUsers, epochs, optimizer, criterion, fraction,
            testloader, learning_rate_server, train_mode, keepProbAvail, keepProbNotAvail, 
            bufferLimit, theta_inner, unit_gradients, adam, temp, cos_similarity_type
            )

        for timeframe in range(num_timeframes):
            print(f"******** Timeframe {timeframe + 1} ********")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
            # Run the FL mode and get updated weights
            new_weights = fl_system.run(run, seed_index, timeframe)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
            if (timeframe%(max(num_users//5,1)) == 0): 
                # Updating the global model with the new aggregated weights 
                with torch.no_grad():
                    for param, saved in zip(model.parameters(), new_weights):
                        param.copy_(saved) 

            
                per_label_accuracy, accuracy = evaluate_per_label_accuracy(model, testloader, device, num_classes=10)

            for index, user in enumerate(fl_system.selected_users_UL):
                chosen_users_over_time[run][seed_index][timeframe][user] = fl_system.selected_users_UL[index]

            accuracy_distributions[run][seed_index][timeframe] = accuracy

            torch.cuda.empty_cache()

            print(f"Mean Accuracy at Timeframe {timeframe + 1}: {accuracy:.2f}%")
        
        for user in range(num_users):
            contribution_distributions[run][seed_index][user] = fl_system.contribution[user]
        for user in range(num_users):
            expected_gradient_magnitude[run][seed_index][user] = fl_system.expected_gradient_magnitude[user]
        num_send = fl_system.num_send
        del model
        del new_weights
        del fl_system
        torch.cuda.empty_cache()

# Prepare data for saving
end_time = time.time()
elapsed_time = end_time - start_time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_data_to_csv(accuracy_distributions, contribution_distributions, chosen_users_over_time, expected_gradient_magnitude, num_users, num_timeframes, args, current_time, start_time, elapsed_time, end_time, num_runs, seeds_for_avg, num_send)

# Relocate the auto-named results folder if an explicit output directory was requested.
if args.out_dir:
    src_dir = f"./results10slot1mem_{current_time}"
    if os.path.isdir(src_dir):
        if os.path.abspath(src_dir) != os.path.abspath(args.out_dir):
            parent = os.path.dirname(os.path.abspath(args.out_dir))
            if parent:
                os.makedirs(parent, exist_ok=True)
            if os.path.exists(args.out_dir):
                shutil.rmtree(args.out_dir)
            shutil.move(src_dir, args.out_dir)
        print(f"Results moved to: {args.out_dir}")
    else:
        print(f"WARNING: results folder '{src_dir}' not found; --out_dir ignored.")