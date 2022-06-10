import importlib
import numpy as np
from ListParameter import ListParameter


base_conifg_str = '''command:
  - ${env}
  - ${interpreter}
  - ${program}
  - --wandb_log
  - ${args}
method: grid
name: ResNet50@NWPU-RESISC45_Final
program: train_lean_model.py
parameters:
  backbone:
    value: ResNet50
  pretraining_dataset:
    value: NWPU-RESISC45
  epochs:
    value: 200
  optimizer:
    value: sgd
  gcn_hidden_units:
    value: 256
'''


gcn_dropout = ListParameter('gcn_dropout', list(np.linspace(0.0,0.9, num=10)), 0.1, 0.2)
lr = ListParameter('lr', [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24], 0.12, 0.16)
#lr_gamma = ListParameter('lr_scheduler_gamma', [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99], 0.8, 0.9)
weight_decay = ListParameter('weight_decay', [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005], 0.0005, 0.001)

#--- iteration 1 ---
gcn_dropout.increase()
lr.decrease()
weight_decay.decrease()

#--- iteration 2 ---
gcn_dropout.decrease_max()
gcn_dropout.decrease()
lr.decrease_max()
lr.decrease()
weight_decay.decrease_max()
weight_decay.decrease()

#--- iteration 3 ---
gcn_dropout.increase_min()
gcn_dropout.increase()
lr.decrease_max()
lr.decrease()
# weight_decay._increase_max()
weight_decay.increase_min()
weight_decay.increase()


#--- iteration 4 ---
gcn_dropout.increase_min()
gcn_dropout.increase()
lr.decrease_max()
lr.decrease()

#--- iteration 5 ---
gcn_dropout.increase_min()
gcn_dropout.increase()
lr.decrease_max()
lr.decrease()
weight_decay.increase_min()
weight_decay.increase()

#--- iteration 5 ---
# DONE!


# #--- Random Seeds ---
gcn_dropout.increase_min()
gcn_dropout.decrease_max()
lr.increase_min()
lr.decrease_max()
weight_decay.increase_min()
weight_decay.decrease_max()

random_seed = ListParameter('random_seed', list(np.linspace(42,71, num=30, dtype=int)), 42, 71)

base_conifg_str += f'{random_seed.get_config_str()}'

params = [gcn_dropout, lr, weight_decay]

print(base_conifg_str)
for param in params:
    print(f'{param.get_config_str()}')




