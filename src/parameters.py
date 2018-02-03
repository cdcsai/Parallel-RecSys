data_param = {
    "nb_users": 6040,
    "nb_movies": 3952,
    "df_size": 1000209,
    "test_ratio": 0.1,
    "data_dir": "data/ratings1M.dat"
}

train_param = {
    "regularization": 0.01,
    "max_iter": 1000,
    "k": 5,
    "learning_rate": 0.01,
    "eval_it" : 100,
    "batch_size": None,
    "is_stochastic": None,
    "processes": 4
}
