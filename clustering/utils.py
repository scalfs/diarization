import easydict

def parse_arguments():
    model_args = easydict.EasyDict({
        "observation_dim": 256,
        "rnn_hidden_size": 512,
        "rnn_depth": 1,
        "rnn_dropout": 0.2,
        "transition_bias": None,
        "crp_alpha": 1.0,
        "sigma2": None,
        "verbosity": 2,
        "enable_cuda": True,
    })
    training_args = easydict.EasyDict({
        "optimizer": 'adam',
        "learning_rate": 1e-3,
        "train_iteration": 1000,
        "batch_size": 10,
        "num_permutations": 10,
        "sigma_alpha": 1.0,
        "sigma_beta": 1.0,
        "regularization_weight": 1e-5,
        "grad_max_norm": 5.0,
        "enforce_cluster_id_uniqueness": True,
    })
    inference_args = easydict.EasyDict({
        "beam_size": 10,
        "look_ahead": 1,
        "test_iteration": 2,
    })

    return (model_args, training_args, inference_args)