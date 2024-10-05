import os
import shutil
import random

original_dataset_dir_cat = 'data/cat'
original_dataset_dir_dog = 'data/dog'

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

if not os.path.exists(train_dir):
    os.makedirs(os.path.join(train_dir, 'cats'))
    os.makedirs(os.path.join(train_dir, 'dogs'))
    
if not os.path.exists(val_dir):
    os.makedirs(os.path.join(val_dir, 'cats'))
    os.makedirs(os.path.join(val_dir, 'dogs'))

split_ratio = 0.8

def split_data(src_dir, train_dest_dir, val_dest_dir, split_ratio=0.8):
    all_files = os.listdir(src_dir)
    
    random.shuffle(all_files)
    
    split_index = int(len(all_files) * split_ratio)
    
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]
 
    for file in train_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(train_dest_dir, file))
    
    for file in val_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(val_dest_dir, file))

split_data(original_dataset_dir_cat, os.path.join(train_dir, 'cats'), os.path.join(val_dir, 'cats'), split_ratio)

split_data(original_dataset_dir_dog, os.path.join(train_dir, 'dogs'), os.path.join(val_dir, 'dogs'), split_ratio)

print("Dataset split complete. Check 'data/train' and 'data/val' directories.")
