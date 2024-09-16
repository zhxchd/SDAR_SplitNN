import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import argparse
import sys
sys.path.append('../src')
import data
from sdar.sdar import SDARAttacker
from sdar.sdar_hetero import SDARHeteroAttacker
from util.util import plot_attack_results, load_config

parser = argparse.ArgumentParser(description='Run SDAR experiment with specified configurations.')
parser.add_argument("setting", type=str, choices=["vsl", "usl"])
parser.add_argument("model", type=str, choices=["resnet", "plainnet"])
parser.add_argument("dataset", type=str, choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])
parser.add_argument("level", type=int, choices=[4,5,6,7,8,9])
parser.add_argument("run", type=int, help="Run number.")

parser.add_argument("--width", type=str, default="standard", choices=["standard", "wide", "narrow"])
parser.add_argument("--aux_data_frac", type=float, default=1.0)
parser.add_argument("--num_class_to_remove", type=int, default=0)
parser.add_argument("--diff_simulator", action="store_true")

parser.add_argument("--ablation", type=str, choices=["no_e_dis", "no_d_dis", "no_dis", "no_cond", "no_label_flip", "naive_sda"], default=None)

# number of heterogenous distributions of the client data
parser.add_argument("--num_hetero_client", type=int, default=1)

# the following arguments are used to specify the defense method
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--alpha", type=float, default=0.0)

parser.add_argument("--print_to_stdout", action="store_true")

args = parser.parse_args()

if sum([args.level > 7, args.l1 != 0.0, args.l2 != 0.0, args.dropout != 0.0, args.alpha != 0.0]) > 1:
        raise ValueError("At most one of l1, l2, dropout, alpha can be not default.")

defense = sum([args.level > 7, args.l1 != 0.0, args.l2 != 0.0, args.dropout != 0.0, args.alpha != 0.0]) > 0

# at most one of width and aux_data_frac can be not default
if sum([args.width != "standard", args.aux_data_frac != 1.0, args.diff_simulator, args.num_class_to_remove != 0, args.ablation != None, args.num_hetero_client > 1, defense]) > 1:
    raise ValueError("Only one of width, aux_data_frac, diff_simulator, num_class_to_remove, ablation, num_hetero_client, defense can be not default.")

# get the directory of this file
curr_dir = os.path.join(os.path.dirname(__file__))

if args.width != "standard":
    rela_dir = f"model_width_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_{args.width}_l{args.level}_run{args.run}"
elif args.aux_data_frac != 1.0:
    rela_dir = f"data_frac_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_{args.aux_data_frac}_l{args.level}_run{args.run}"
elif args.diff_simulator:
    rela_dir = f"diff_simulator_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"
elif args.num_class_to_remove != 0:
    rela_dir = f"remove_aux_class_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_aux_wo_{args.num_class_to_remove}_l{args.level}_run{args.run}"
elif args.ablation != None:
    rela_dir = f"ablation_results/{args.ablation}/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"
elif args.num_hetero_client > 1:
    rela_dir = f"hetero_clients_results/{args.setting}/{args.dataset}"
    file_name = f"{args.num_hetero_client}_{args.model}_l{args.level}_run{args.run}"
elif defense:
    if args.l1 != 0.0:
        rela_dir = f"defense_results/l1/{args.setting}/{args.dataset}"
        file_name = f"{args.l1}_{args.model}_l{args.level}_run{args.run}"
    elif args.l2 != 0.0:
        rela_dir = f"defense_results/l2/{args.setting}/{args.dataset}"
        file_name = f"{args.l2}_{args.model}_l{args.level}_run{args.run}"
    elif args.dropout != 0.0:
        rela_dir = f"defense_results/dropout/{args.setting}/{args.dataset}"
        file_name = f"{args.dropout}_{args.model}_l{args.level}_run{args.run}"
    elif args.alpha != 0.0:
        rela_dir = f"defense_results/decorrelation/{args.setting}/{args.dataset}"
        file_name = f"{args.alpha}_{args.model}_l{args.level}_run{args.run}"
    elif args.level > 7:
        rela_dir = f"defense_results/deeper_f/{args.setting}/{args.dataset}"
        file_name = f"{args.model}_l{args.level}_run{args.run}"
else:
    rela_dir = f"sdar_results/{args.setting}/{args.dataset}"
    file_name = f"{args.model}_l{args.level}_run{args.run}"

base_dir = os.path.join(curr_dir, rela_dir)
fig_dir = os.path.join(base_dir, "fig")
output_dir = os.path.join(base_dir, "output")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not args.print_to_stdout:
    sys.stdout = open(os.path.join(base_dir, file_name + ".txt"),'wt')

u_shape = (args.setting == "usl")
conditional = not u_shape
num_class = 100 if args.dataset == "cifar100" else 200 if args.dataset == "tinyimagenet" else 10
batch_size = 32 if args.dataset == "stl10" else 128 # since stl10 contains only 13000 images
num_iter = 40000 if args.level > 7 else 20000 if args.dataset in ["cifar10", "cifar100", "tinyimagenet", "stl10"] else 10000
config = load_config("sdar", args.model, args.dataset, u_shape)

if args.ablation == "no_e_dis":
    config["lambda1"] = 0.0
elif args.ablation == "no_d_dis":
    config["lambda2"] = 0.0
elif args.ablation == "no_dis":
    config["lambda1"] = 0.0
    config["lambda2"] = 0.0
elif args.ablation == "no_cond":
    conditional = False
elif args.ablation == "no_label_flip":
    config["flip_rate"] = 0.0
elif args.ablation == "naive_sda":
    config["lambda1"] = 0.0
    config["lambda2"] = 0.0
    config["flip_rate"] = 0.0
    conditional = False

if args.num_hetero_client == 1:
    client_ds, server_ds = data.load_dataset(args.dataset, frac=args.aux_data_frac, num_class_to_remove=args.num_class_to_remove)
else:
    client_ds_list, server_ds_list = data.load_dataset_with_hetero_clients(args.dataset, args.num_hetero_client)

print(f"Start {args.setting}/{args.model} experiments on {args.dataset} at l{args.level} with {config}")

if args.num_hetero_client == 1:
    sdar_attacker = SDARAttacker(client_ds, server_ds, num_class=num_class, batch_size=batch_size)
else:
    sdar_attacker = SDARHeteroAttacker(client_ds_list, server_ds_list, num_class=num_class, batch_size=batch_size)

start_time = time.time()
history = sdar_attacker.run(level=args.level, num_iter=num_iter, config=config, u_shape=u_shape, conditional=conditional, model_type=args.model, width=args.width, diff_simulator=args.diff_simulator, l1=args.l1, l2=args.l2, dropout=args.dropout, alpha=args.alpha, verbose_freq=100)
end_time = time.time()

print(f"Experiment took {end_time - start_time} seconds, on average {(end_time - start_time)/num_iter} seconds/iteration.")
print("Experiment done. Saving history...")

if args.num_hetero_client == 1:
    _ = sdar_attacker.evaluate(verbose=True)
else:
    eval_ds, _ = data.load_dataset(args.dataset, frac=args.aux_data_frac, num_class_to_remove=args.num_class_to_remove)
    _ = sdar_attacker.evaluate(eval_ds, verbose=True)

eval_batch = data.load_single_batch(args.dataset, batch_size=batch_size)
x,y = [(x,y) for (x,y) in iter(eval_batch)][0]

x_recon, mse = sdar_attacker.attack(x, y)
t = 10
plot_attack_results(x_recon[t:t+20], os.path.join(fig_dir, file_name + ".png"))
example_mse = np.mean((x[t:t+20] - x_recon[t:t+20])**2)
print(f"Attack MSE on batch: {mse}, on these examples: {example_mse}")

np.save(os.path.join(output_dir, file_name + ".npy"), history)
print(f"History saved to {os.path.join(rela_dir, file_name + '.npy')}")
