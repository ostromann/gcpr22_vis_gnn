from email.mime import base
import numpy as np
from ListParameter import ListParameter

base_conifg_str = '''command:
  - ${env}
  - ${interpreter}
  - ${program}
  - --wandb_log
  - ${args}
method: grid
name: Hist
program: train_lean_model_hist.py
parameters:
  hist_bins:
    value: 32
  epochs:
    value: 200
  optimizer:
    value: sgd
  gcn_hidden_units:
    value: 256
'''


gcn_dropout = ListParameter('gcn_dropout', list(np.linspace(0.0,0.9, num=10)), 0.1, 0.2)
lr = ListParameter('lr', [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36], 0.12, 0.16)
weight_decay = ListParameter('weight_decay', [0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005], 0.0005, 0.001)


#--- iteration 1 ---
gcn_dropout.decrease()
lr.decrease()
weight_decay.decrease()


#--- iteration 2 ---
gcn_dropout.increase()
gcn_dropout.increase_min()
lr.increase()
lr.increase_min()
weight_decay.decrease_max()
weight_decay.decrease()

#--- iteration 3 ---
lr.increase()
lr.increase_min()
weight_decay.decrease_max()
weight_decay.decrease()


#--- iteration 4 ---


#--- iteration 5 ---


#--- iteration 6 ---
# DONE!


# #--- iteration 6 ---
# lr.increase_min()
# lr.increase()
# gcn_dropout.increase_min()
# gcn_dropout.increase()

# #--- iteration 7 ---
# lr.increase_min()
# lr.increase()

#--- Random Seeds ---
gcn_dropout.increase_min()
gcn_dropout.decrease_max()
lr.increase_min()
lr.decrease_max()
weight_decay.increase_min()
weight_decay.decrease_max()

random_seed = ListParameter('random_seed', list(np.linspace(42,71, num=30, dtype=int)), 42, 71)

base_conifg_str += f'{random_seed.get_config_str()}'



params = [gcn_dropout, lr, weight_decay] #lr_gamma,

print(base_conifg_str)
for param in params:
    print(f'{param.get_config_str()}')




