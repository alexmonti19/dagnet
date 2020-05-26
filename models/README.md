# Models

The repo contains both the final model and the two ablation architectures cited in the paper.
- *VRNN*: the baseline generative architecture
- *A-VRNN*: enhanced version of the baseline with a single graph (on agents' hiddens)
- *DAG-Net*: the complete architecture with two graphs

Before launching the training scripts, make sure to have downloaded and preprocessed the datasets. See 
[datasets/README.md](../datasets/README.md).





## Baseline
The baseline is an implementation of the Variational Recurrent Neural Network from 
[https://arxiv.org/pdf/1506.02216.pdf](https://arxiv.org/pdf/1506.02216.pdf).  

### Training [bsk]
```
python train_bsk.py --players [atk | def | all] --run <expname>
```

Main arguments:
- `--obs_len`: Time-steps of observation. Default is '10'.
- `--pred_len`: Time-steps of prediction. Default is '40'.
- `--players`: On which players the model has to focus. Choices: ``atk`` | ``def`` | ``all``.
###
- `--learning_rate`: Initial learning rate.
- `--lr_scheduler`: Use the learning rate scheduling policy (ReduceOnPlateau). Default is 'False'.
- `--warmup`: Warm-up the KLD in the early training epochs. Default is 'False'.
- `--batch_size`: Mini-batch size.
- `--num_epochs`: Training epochs.
###
- `--n_layers`: Number of recurrent layers.
- `--x_dim`: Input examples size.
- `--h_dim`: Dimension of the hidden layers.
- `--z_dim`: Dimension of the latents variables.
- `--rnn_dim`: Dimensions of the hidden states of the recurrent cell(s).
###
- `--save_every`: Periodic checkpoint saving.
- `--eval_every`: Periodic model evaluation.
- `--num_samples`: Number of samples to evaluate model.
- `--run`: Experiment name.
- `--resume`: Resume from last saved checkpoint.

### Training [sdd]
```
python train_sdd.py --run <expname>
```

Main arguments:
- `--obs_len`: Time-steps of observation. Default is '8'.
- `--pred_len`: Time-steps of prediction. Default is '12'.
###
- `--learning_rate`: Initial learning rate.
- `--lr_scheduler`: Use the learning rate scheduling policy (ReduceOnPlateau). Default is 'False'.
- `--warmup`: Warm-up the KLD in the early training epochs. Default is 'False'.
- `--batch_size`: Mini-batch size.
- `--num_epochs`: Training epochs.
###
- `--n_layers`: Number of recurrent layers.
- `--x_dim`: Input examples size.
- `--h_dim`: Dimension of the hidden layers.
- `--z_dim`: Dimension of the latents variables.
- `--rnn_dim`: Dimensions of the hidden states of the recurrent cell(s).
###
- `--save_every`: Periodic checkpoint saving.
- `--eval_every`: Periodic model evaluation.
- `--num_samples`: Number of samples to evaluate model.
- `--run`: Experiment name.
- `--resume`: Resume from last saved checkpoint.






## Attentive VRNN
An enhanced version of the original VRNN: we apply a first graph refinement step on single agents' hidden states. 
The refinement allows to share past motion behavioural information across all the agents inside a given scene.  

### Training [bsk]
```
python train_bsk.py --players [atk | def | all] --graph_model [gat | gcn] --run <expname>
```

Main arguments:
- `--obs_len`: Time-steps of observation. Default is '10'.
- `--pred_len`: Time-steps of prediction. Default is '40'.
- `--players`: On which players the model has to focus. Choices: ``atk`` | ``def`` | ``all``.
###
- `--learning_rate`: Initial learning rate
- `--lr_scheduler`: Use the learning rate scheduling policy (ReduceOnPlateau). Default is 'False'.
- `--warmup`: Warm-up the KLD in the early training epochs. Default is 'False'.
- `--batch_size`: Mini-batch size.
- `--num_epochs`: Training epochs.
###
- `--n_layers`: Number of recurrent layers.
- `--x_dim`: Input examples size.
- `--h_dim`: Dimension of the hidden layers.
- `--z_dim`: Dimension of the latents variables.
- `--rnn_dim`: Dimensions of the hidden states of the recurrent cell(s).
###
- `--graph_model`: Which type of Graph Neural Network to use. Choices: ``gat`` | ``gcn``.
- `--graph_hid`: Graph hidden layer dimension. Default is '8'.
- `--adjacency_type`: Type of adjacency matrix to describe the graph relations inside a scene. Choices: ``0`` 
(binary fully-connected), ``1`` (distance similarity matrix), ``2`` (KNN similarity matrix). Default is '1'.
- `--top_k_neigh`: Number of neighbours to consider when using the KNN similarity matrix.
- `--n_heads`: Number of heads to use when employing GAT. Default is '4'.
- `--alpha`: Negative step for the LeakyReLU activation inside GAT.
###
- `--save_every`: Periodic checkpoint saving.
- `--eval_every`: Periodic model evaluation.
- `--num_samples`: Number of samples to evaluate model.
- `--run`: Experiment name.
- `--resume`: Resume from last saved checkpoint.

### Training [sdd]
```
python train_sdd.py --graph_model [gat | gcn] --run <expname>
```

Main arguments:
- `--obs_len`: Time-steps of observation. Default is '8'.
- `--pred_len`: Time-steps of prediction. Default is '12'.
###
- `--learning_rate`: Initial learning rate.
- `--lr_scheduler`: Use the learning rate scheduling policy (ReduceOnPlateau). Default is 'False'.
- `--warmup`: Warm-up the KLD in the early training epochs. Default is 'False'.
- `--batch_size`: Mini-batch size.
- `--num_epochs`: Training epochs.
###
- `--n_layers`: Number of recurrent layers.
- `--x_dim`: Input examples size.
- `--h_dim`: Dimension of the hidden layers.
- `--z_dim`: Dimension of the latents variables.
- `--rnn_dim`: Dimensions of the hidden states of the recurrent cell(s).
###
- `--graph_model`: Which type of Graph Neural Network to use. Choices: ``gat`` | ``gcn``.
- `--graph_hid`: Graph hidden layer dimension. Default is '8'.
- `--adjacency_type`: Type of adjacency matrix to describe the graph relations inside a scene. Choices: ``0`` 
(binary fully-connected), ``1`` (distance similarity matrix), ``2`` (KNN similarity matrix). Default is '1'.
- `--top_k_neigh`: Number of neighbours to consider when using the KNN similarity matrix. Default is '3'.
- `--n_heads`: Number of heads to use when employing GAT. Default is '4'.
- `--alpha`: Negative step for the LeakyReLU activation inside GAT.
###
- `--save_every`: Periodic checkpoint saving.
- `--eval_every`: Periodic model evaluation.
- `--num_samples`: Number of samples to evaluate model.
- `--run`: Experiment name.
- `--resume`: Resume from last saved checkpoint.





## DAG-Net
The final model: as before, we apply a graph refinement step on single agents' hidden states to share past information. 
This step is flanked by another graph refinement step on agents' goals, expressed as areas/cells of the environment
where agents will land in the future. 

### Training [bsk]
```
python train_bsk.py --players [atk | def | all] --graph_model [gat | gcn] --run <expname>
```

Main arguments:
- `--obs_len`: Time-steps of observation. Default is '10'.
- `--pred_len`: Time-steps of prediction. Default is '40'.
- `--players`: On which players the model has to focus. Choices: ``atk`` | ``def`` | ``all``.
###
- `--learning_rate`: Initial learning rate.
- `--lr_scheduler`: Use the learning rate scheduling policy (ReduceOnPlateau). Default is 'False'.
- `--warmup`: Warm-up the KLD in the early training epochs. Default is 'False'.
- `--CE_weight`: Cross-entropy weight for the training of the goals sampling module. Default is '1e-2'.
- `--batch_size`: Mini-batch size.
- `--num_epochs`: Training epochs.
###
- `--n_layers`: Number of recurrent layers.
- `--x_dim`: Input examples size.
- `--h_dim`: Dimension of the hidden layers.
- `--z_dim`: Dimension of the latents variables.
- `--rnn_dim`: Dimensions of the hidden states of the recurrent cell(s).
###
- `--graph_model`: Which type of Graph Neural Network to use. Choices: ``gat`` | ``gcn``.
- `--graph_hid`: Graph hidden layer dimension. Default is '8'.
- `--adjacency_type`: Type of adjacency matrix to describe the graph relations inside a scene. Choices: ``0`` 
(binary fully-connected), ``1`` (distance similarity matrix), ``2`` (KNN similarity matrix). Default is '1'.
- `--top_k_neigh`: Number of neighbours to consider when using the KNN similarity matrix. Default is '3'.
- `--n_heads`: Number of heads to use when employing GAT. Default is '4'.
- `--alpha`: Negative step for the LeakyReLU activation inside GAT.
###
- `--save_every`: Periodic checkpoint saving.
- `--eval_every`: Periodic model evaluation.
- `--num_samples`: Number of samples to evaluate model.
- `--run`: Experiment name.
- `--resume`: Resume from last saved checkpoint.

### Training [sdd]
```
python train_sdd.py --graph_model [gat | gcn] --run <expname>
```

Main arguments:
- `--obs_len`: Time-steps of observation. Default is '8'.
- `--pred_len`: Time-steps of prediction. Default is '12'.
- `--goals_window`: How many time-steps . Default is '3'.
- `--n_cells_x`: Number of cells (goals) along the x dimension. Default is '32'.
- `--n_cells_y`: Number of cells (goals) along the y dimension. Default is '30'.
###
- `--learning_rate`: Initial learning rate.
- `--lr_scheduler`: Use the learning rate scheduling policy (ReduceOnPlateau). Default is 'False'.
- `--warmup`: Warm-up the KLD in the early training epochs. Default is 'False'.
- `--CE_weight`: Cross-entropy weight for the training of the goals sampling module. Default is '1
- `--batch_size`: Mini-batch size.
- `--num_epochs`: Training epochs.
###
- `--n_layers`: Number of recurrent layers.
 `--x_dim`: Input examples size.
- `--h_dim`: Dimension of the hidden layers.
- `--z_dim`: Dimension of the latents variables.
- `--rnn_dim`: Dimensions of the hidden states of the recurrent cell(s).
###
- `--graph_model`: Which type of Graph Neural Network to use. Choices: ``gat`` | ``gcn``.
- `--graph_hid`: Graph hidden layer dimension. Default is '8'.
- `--adjacency_type`: Type of adjacency matrix to describe the graph relations inside a scene. Choices: ``0`` 
(binary fully-connected), ``1`` (distance similarity matrix), ``2`` (KNN similarity matrix). Default is '1'.
- `--top_k_neigh`: Number of neighbours to consider when using the KNN similarity matrix. Default is '3'.
- `--n_heads`: Number of heads to use when employing GAT. Default is '4'.
- `--alpha`: Negative step for the LeakyReLU activation inside GAT.
###
- `--save_every`: Periodic checkpoint saving.
- `--eval_every`: Periodic model evaluation.
- `--num_samples`: Number of samples to evaluate model.
- `--run`: Experiment name.
- `--resume`: Resume from last saved checkpoint.




## Evaluations
To evaluate the results once completed the training phase, each of the previous models has its own .py scripts in its 
directory. Move in the specific model directory and run:
```
python evaluate_bsk.py --run <expname> --best
```
or
```
python evaluate_sdd.py --run <expname> --best
```

Once having specified the experiment to evaluate with the ``--run`` argument (N.B. ``<expname>`` is the name of 
the experiment directory inside ``\runs\<model>``), the script will automatically look for the relative checkpoint. The user have 
three options:
- Load the best checkpoint, by using the ``--best`` flag (as above)
- Load a checkpoint from a specific epoch, by using the ``--epoch <epoch_number>`` argument
- Let the model pick the last saved checkpoint, by specifying nothing
