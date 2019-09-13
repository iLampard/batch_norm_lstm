# batch_norm_lstm


In an attempt to learn Tensorflow, I have implemented 
[Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025)
 for the pixel-by-pixel MNIST classification using Tensorflow 1.13.
- A batch normalization operation for each time step has been implemented based on 
the [discussion](https://github.com/OlavHN/bnlstm/issues/7).
- A *ModelRunner* class is added to control the pipeline of model 
training and evaluation.

## How to run

Suppose we want to run 50 epochs and use Tensorboard to 
visualize the process

```bash
cd bn_lstm
python main.py --write_summary True --max_epoch 50
```

To check the description of all flags
```bash
python main.py -helpful
```

To open tensorboard
```bash
tensorboard --logdir=path
```

where *path* can be found in the log which shows the relative dir where the model is saved, e.g. 
*logs/ModelWrapper/lr-0.001_dim-32/20190912-230850/saved_model/tfb_dir*.

## Requirement

```bash
tensorflow==1.13.1
```



# Reference
- [Github - bnlstm](https://github.com/OlavHN/bnlstm)