import os
import shutil

data_dir = 'data/WhatsApp'
out_dir = 'data'

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        src = os.path.join(folder_path, '_chat.txt')
        dst = os.path.join(out_dir, f'{folder}.txt')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'Copied {src} to {dst}')
        else:
            print(f'Skipped {folder}: _chat.txt not found')
