from matplotlib import pyplot as plt

def plot_attack_results(X, file_name):
    n = len(X)
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(X[i])
        ax[i].set(xticks=[], yticks=[])
        ax[i].set_aspect('equal')
    plt.savefig(file_name, dpi=fig.dpi, bbox_inches='tight')
    return fig

def load_config(method, model, dataset, u_shape, diff_simulator=False):
    if method == "unsplit":
        return {
            "x_lr": 0.001,
            "e_lr": 0.001,
            "lambda": 1.0,
            "max_iter": 1000,
            "max_x_iter": 100,
            "max_e_iter": 100
        }
    elif method == "fsha":
        return {
            "gen_lr": 1e-5,
            "dis_lr": 1e-4,
            "ed_lr": 1e-5,
            "gp": 500.0,
        }
    
    config = {"e_lr": 0.001, "d_lr": 0.0005}

    if u_shape:
        config["hs_lr"] = 0.001

    if method == "pcat":
        return config
    
    if (not u_shape):
        config["lambda1"] = 0.02
        config["lambda2"] = 1e-5
    else:
        config_table = {
            "resnet": {
                "cifar10": {"lambda1": 0.02, "lambda2": 1e-5, "flip_rate": 0.2},
                "cifar100": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.2},
                "tinyimagenet": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.2},
                "stl10": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.2}, # no idea, just copy the tinyimagenet one
            },
            "plainnet": {
                "cifar10": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.1},
                "cifar100": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.1},
                "tinyimagenet": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.4},
                "stl10": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.4},  # no idea, just copy the tinyimagenet one
            }
        }

        config = {**config, **config_table[model][dataset]}
    
    config["e_dis_lr"] = config["e_lr"] * config["lambda1"]
    config["d_dis_lr"] = config["d_lr"] * config["lambda2"]

    return config