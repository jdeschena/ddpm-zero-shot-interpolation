import numpy as np
import torch as th
from torch.utils.data import DataLoader
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import argparse
from diff_exp.utils import (
    add_args,
    seed_everything,
    default_arguments as cli_defaults,
    save_args_to_cfg,
    parse_args,
    mkdirs4file,
)
from diff_exp.data.from_npz_dataset import NPZDataset
from omegaconf import OmegaConf
from tqdm import tqdm


def default_args():
    return dict(
        model_name="openai/clip-vit-base-patch32",
        batch_size=4,
        device="cpu",
        input_npz="/home/anon/samples/uncond_celeb_all_ddpm/samples_10000x64x64x3.npz",
        output_npz="default.npz",
        num_workers=1,
    )


def load_model_and_processor(model_name):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


class ProcessorDataset(th.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.processor(images=x, return_tensors="pt", padding=True)
        x["pixel_values"] = x["pixel_values"].squeeze(0)

        return x

    def __len__(self):
        return len(self.dataset)


def get_image_embeddings(args, model, processor, dataloader):
    embeddings = []
    for x in tqdm(dataloader, desc="Embedding data..."):
        x = x.to(args.device)
        outputs = model.get_image_features(**x)
        outputs = outputs.cpu()
        embeddings.append(outputs)
    
    embeddings = th.cat(embeddings, dim=0)
    return embeddings


def main(args):
    th.set_grad_enabled(False)
    model, processor = load_model_and_processor(args.model_name)
    data = NPZDataset([args.input_npz])
    data = ProcessorDataset(data, processor)

    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers)
    model = model.to(args.device)
    model.eval()

    all_embeddings = get_image_embeddings(args, model, processor, loader)
    mkdirs4file(args.output_npz)
    np.savez(args.output_npz, all_embeddings)
    print(f"Saved {len(all_embeddings)} embeddings to {args.output_npz}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser, cli_defaults())
    add_args(parser, default_args())

    args = parse_args(parser)
    save_args_to_cfg(args)

    # Small logging:
    print("### Arguments for the run: ###")
    print(OmegaConf.to_yaml(args, sort_keys=True))
    print("-----")
    # Seed things
    seed_everything(args.seed)
    main(args)
