from diff_exp.data.attribute_celeba_dataset import Dataset, default_args as data_args
from diff_exp.utils import mkdirs4file, TransformDataset
from diff_exp.transforms_utils import get_transform
from omegaconf import OmegaConf
from tqdm import tqdm
from diff_exp.models.efficientnet import default_args as model_args, get_model
import torch as th
import yaml
import os


def pred_all(model, dataset, batch_size, device):
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=max(int(os.cpu_count() // 2), 1),
        shuffle=False,
    )

    model.eval()
    model = model.to(device)
    all_preds = []

    for batch in tqdm(loader, desc="Predict..."):
        x, y = batch
        x = x.to(device)
        with th.inference_mode():
            x = model(x)
        x = x.cpu()
        all_preds.append(x)
    all_preds = th.concat(all_preds, dim=0)
    return all_preds



args = model_args()
args = OmegaConf.create(args)
print(OmegaConf.to_yaml(args))
# Load model
model = get_model(args)
ckpt_path = "/home/anon/artifacts/test_classifiers/smile_64_64_cls_longer.ckpt"
ckpt = th.load(ckpt_path, map_location="cpu")['state_dict']

out_dict = dict()
for k, v in ckpt.items():
    k = k.replace("model.", "")
    out_dict[k] = v
    

model.load_state_dict(out_dict)


# Load data
ds_args = data_args()
ds_args = OmegaConf.create(ds_args)
ds_args.data_dir = "../../data"
ds_args.filter_path = "../../data/smile_filtered_5_confident_09999/train.txt"
dataset = Dataset(**ds_args)


transform_str = """
- - to_tensor
- - center_crop
  - size: 178
- - resize
  - size: 64
- - normalize
  - mean: 0.5, 0.5, 0.5
  - std: 0.5, 0.5, 0.5
""".strip()

transform_str = yaml.load(transform_str, yaml.Loader)
transform = get_transform(transform_str)

dataset = TransformDataset(dataset, transform)


all_preds = pred_all(model, dataset, 64, "cuda:0").softmax(-1)

pairs = [(idx, float(p)) for idx, p in enumerate(all_preds[:, 1])]
pairs.sort(key=lambda x: x[1])
in_subset_idxs = [idx for idx, _ in pairs]

map_idx = dataset.dataset.idxs
def take_top(arr, map_idx, n):
    split = arr[:n] + arr[-n:]
    split = [map_idx[idx] for idx in split]
    split.sort()
    return split

def save(idxs, name):
    with open(name, "w") as f:
        line = "\n".join((str(x) for x in idxs))
        f.write(line)

split_60k = take_top(in_subset_idxs, map_idx, 30_000)
split_30k = take_top(in_subset_idxs, map_idx, 15_000)
split_10k = take_top(in_subset_idxs, map_idx, 5_000)
split_5k = take_top(in_subset_idxs, map_idx, 2_500)


print("Len 60k split:", len(split_60k))
print("Len 30k split:", len(split_30k))
print("Len 10k split:", len(split_10k))
print("Len 5k split:", len(split_5k))

save(split_60k, "60k.txt")
save(split_30k, "30k.txt")
save(split_10k, "10k.txt")
save(split_5k, "5k.txt")

# Then run cp *.txt ../../data/celeba_smile_train_size_ablation/