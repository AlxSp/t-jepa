#%%
import os
import shutil

loss_dir = 'losses'

run_loss_file_path = os.path.join('out', 'losses.jsonl')
batch_file = os.path.join('out', 'batch.jsonl')
#%%
train_script = 'train.py'
train_scenarios = ['scratch', 'resume']
main_iterations = 12
train_iterations = [main_iterations, main_iterations // 2]
#%%
# create loss directory if it does not exist
if os.path.exists(loss_dir):
    shutil.rmtree(loss_dir)

os.mkdir(loss_dir)

# run train script 
os.system(f'python {train_script} {train_scenarios[0]} {train_iterations[0]}')
# move loss file to loss directory and rename it to losses_0.jsonl
shutil.move(run_loss_file_path, os.path.join(loss_dir, f'losses_0.jsonl'))
# move batch file to loss directory and rename it to batch_0.jsonl
shutil.move(batch_file, os.path.join(loss_dir, f'batch_0.jsonl'))

# run train script again with same parameters
os.system(f'python {train_script} {train_scenarios[0]} {train_iterations[1]}')
# move loss file to loss directory and rename it to losses_1.jsonl
shutil.copy(run_loss_file_path, os.path.join(loss_dir, f'losses_1.jsonl'))
# move batch file to loss directory and rename it to batch_1.jsonl
shutil.copy(batch_file, os.path.join(loss_dir, f'batch_1.jsonl'))
#%%
# double check that the two batch files are identical
with open(os.path.join(loss_dir, 'batch_0.jsonl'), 'r') as f:
    batch_0 = f.readlines()

with open(os.path.join(loss_dir, 'batch_1.jsonl'), 'r') as f:
    batch_1 = f.readlines()

assert batch_0[:len(batch_0) // 2] == batch_1, 'Batch files are not identical'

print('Batch 0 1/2 and Batch 1 are identical')
# double check that the two loss files are identical
#%%
with open(os.path.join(loss_dir, 'losses_0.jsonl'), 'r') as f:
    losses_0 = f.readlines()

with open(os.path.join(loss_dir, 'losses_1.jsonl'), 'r') as f:
    losses_1 = f.readlines()

assert losses_0[:len(losses_0) // 2] == losses_1, 'Loss files are not identical'

print('Losses 0 1/2 and Losses 1 are identical')
print('Test passed')
print('\n\n')
#%%
with open(run_loss_file_path, 'r') as file:
    lines = file.readlines()
with open(run_loss_file_path, 'w') as file:
    file.write(''.join(lines[:-1]))

with open(batch_file, 'r') as file:
    lines = file.readlines()
with open(batch_file, 'w') as file:
    file.write(''.join(lines[:-1]))


# run train script again with same parameters but with resume scenario
os.system(f'python {train_script} {train_scenarios[1]} {train_iterations[0]}')

# move loss file to loss directory and rename it to losses_2.jsonl
shutil.move(run_loss_file_path, os.path.join(loss_dir, f'losses_2.jsonl'))
# move batch file to loss directory and rename it to batch_2.jsonl
shutil.move(batch_file, os.path.join(loss_dir, f'batch_2.jsonl'))

# double check that the two batch files are identical
with open(os.path.join(loss_dir, 'batch_0.jsonl'), 'r') as f:
    batch_0 = f.readlines()

with open(os.path.join(loss_dir, 'batch_2.jsonl'), 'r') as f:
    batch_2 = f.readlines()

assert batch_0 == batch_2, 'Batch files are not identical'

# double check that the two loss files are identical
with open(os.path.join(loss_dir, 'losses_0.jsonl'), 'r') as f:
    losses_0 = f.readlines()

with open(os.path.join(loss_dir, 'losses_2.jsonl'), 'r') as f:
    losses_2 = f.readlines()

assert losses_0 == losses_2, 'Loss files are not identical'

