from pathlib import Path
import argparse
from PIL import Image

import torch

from karlo.sampler.i2i import I2ISampler
from karlo.utils.util import set_seed


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True, help='path for model checkpoints')
    parser.add_argument('--data-dir', type=str, required=True, help='path for data')
    parser.add_argument('--n-samples', type=int, default=-1, help='#images to generate')
    parser.add_argument('--max-bsz', type=int, default=1, help='#Batch size to generate')
    parser.add_argument('--output-dir', type=str, default='outputs', help='output path for generated images')
    parser.add_argument('--sampling-type', type=str, default='fast', choices=('fast', 'default'))
    parser.add_argument('--use_bf16', action='store_true', default=False, help='If true, use bf16 for inference.')
    parser.add_argument('--seed', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    out_path = Path(args.output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    model = I2ISampler.from_pretrained(
        root_dir=args.root_dir,
        clip_model_path="ViT-L-14.pt",
        clip_stat_path="ViT-L-14_stats.th",
        sampling_type=args.sampling_type,
        use_bf16=args.use_bf16,
    )

    data_path = Path(args.data_dir)

    for subfolder in data_path.iterdir():
        if not subfolder.is_dir():
            continue

        print(f'Processing {subfolder.name}...')
        curr_out_path = out_path / subfolder.name
        curr_out_path.mkdir(exist_ok=True, parents=True)

        img_paths = [p for p in subfolder.glob('*') if p.is_file()][:args.n_samples]

        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')

            img_out = iter(model(img, bsz=args.max_bsz, progressive_mode="final")).__next__()
            img_out = (torch.permute(img_out * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy())
            img_out = Image.fromarray(img_out[0])
            img_out.save(curr_out_path / f'{img_path.stem}_{args.seed}.JPEG')
