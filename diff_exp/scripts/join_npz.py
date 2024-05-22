import numpy as np
import os

# TODO: remove 10k file before merging again
# smile conditional
#parent_dir = "/home/anon/samples/cond_smile_celeba_all_ddpm_guid_35"
# non-smile conditional
#parent_dir = "/home/anon/samples/smile_multi_35_35"
# parent_dir = "/home/anon/samples/uncond_celeb_extreme_thr_01_09"
#parent_dir = "/home/anon/samples/multi_guid_smile_extreme_thr_01_09"
parent_dir = "/home/anon/samples/bald_one_multiclassifier_guid_10/"
all_files = []
n_samples = 256
target = f"samples_{n_samples}x64x64x3.npz"


for fname in os.listdir(parent_dir):
    if fname[-4:] == ".npz" and target not in fname:
        arr = np.load(os.path.join(parent_dir, fname))
        arr = arr['arr_0']
        all_files.append(arr)

all_files = np.concatenate(all_files, axis=0) #[:10_000]
print(f"Loaded {len(all_files)} samples. Keeping {n_samples}")
all_files = all_files[:10_000]
save_path = os.path.join(parent_dir, target)
np.savez(save_path, all_files)
print("Save path:", save_path)
