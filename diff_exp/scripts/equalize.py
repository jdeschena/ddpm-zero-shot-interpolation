from diff_exp.data.celeba import get_dataset, default_args
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.multiprocessing
from matplotlib import pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')


def compute_loss(labels, weights, target):
    with th.no_grad():
        th.clamp_(weights, min=1)
    norm_weights = weights / th.sum(weights)
    frac = th.sum(norm_weights * labels, dim=0)
    loss = (frac - target) ** 2
    loss = loss.mean()
    return loss


def step(labels, weights, target, optim):
    loss = compute_loss(labels, weights, target)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return float(loss.detach())


args = default_args()
args = OmegaConf.create(args)
train_set = get_dataset(args, "train")

loader = DataLoader(train_set, batch_size=16, num_workers=12)

#all_targets = []
#for x, y in tqdm(loader):
#    all_targets.append(y)
#all_targets = th.concat(all_targets, dim=0)
#
#th.save(all_targets, "ttt.pt")

all_targets = th.load("ttt.pt")

with th.no_grad():
    weights = th.randn(all_targets.shape[0], 1) + 5
    orig_weights = weights.clone()
target = th.ones(all_targets.shape[1]) / 2

weights = weights.cuda()
all_targets = all_targets.cuda()
target = target.cuda()


weights.requires_grad_(True)

N = 100_000
all_losses = []

opt = th.optim.SGD([weights], lr=10000)
for _ in tqdm(range(N), total=N):
    l = step(all_targets, weights, target, opt)
    all_losses.append(l)



N = 50_000

opt = th.optim.SGD([weights], lr=1000)
for _ in tqdm(range(N), total=N):
    l = step(all_targets, weights, target, opt)
    all_losses.append(l)

N = 25_000

opt = th.optim.SGD([weights], lr=100)
for _ in tqdm(range(N), total=N):
    l = step(all_targets, weights, target, opt)
    all_losses.append(l)

N = 12_500

opt = th.optim.SGD([weights], lr=10)
for _ in tqdm(range(N), total=N):
    l = step(all_targets, weights, target, opt)
    all_losses.append(l)




with th.no_grad():
    norm_weights = weights / th.sum(weights)
    frac = th.sum(norm_weights * all_targets, dim=0)
    print("preds frac")
    print(frac)
    print("vanilla fracs")
    print(th.mean(all_targets.float(), dim=0))
    torch.save(
        norm_weights, 
        "/home/anon/artifacts/is_weights/celeba_train.pt"
    )

plt.plot(all_losses)
plt.savefig("losses.png")

