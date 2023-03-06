import os
import argparse
import logging
import time
from datetime import datetime
from PIL import Image

import torch

from karlo.sampler.text_plus_images import TextPlusImagesSampler
from karlo.utils.util import set_seed


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", type=str, required=True, help="path for model checkpoints"
    )
    parser.add_argument("--n-samples", type=int, default=1, help="#images to generate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="output path for generated images",
    )
    parser.add_argument(
        "--sampling-type",
        type=str,
        default="fast",
        choices=("fast", "default"),
    )
    parser.add_argument("--img-paths", nargs="+", help="Paths to input image to be used for generation.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation.")
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=False,
        help="If true, use bf16 for inference."
    )
    parser.add_argument("--img-weights", nargs="+", type=float, required=False,
                        help="Weights for each image. If not provided, all images and prompt will have the same weight."
                             " Sum of image weights plut text weight must be 1.0.")
    parser.add_argument("--text-weight", type=float, required=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    logging.getLogger().setLevel(logging.INFO)

    save_dir = os.path.join(args.output_dir, datetime.now().strftime("%d%m%Y_%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = TextPlusImagesSampler.from_pretrained(
        root_dir=args.root_dir,
        clip_model_path="ViT-L-14.pt",
        clip_stat_path="ViT-L-14_stats.th",
        sampling_type=args.sampling_type,
        use_bf16=args.use_bf16,
    )

    images = [Image.open(img_path).convert('RGB') for img_path in args.img_paths]
    if args.img_weights is None and args.text_weight is None:
        args.img_weights = [1.0 / (len(images) + 1)] * len(images)
        args.text_weight = 1.0 / (len(images) + 1)
    elif args.img_weights is None:
        args.img_weights = [(1.0 - args.text_weight) / len(images)] * len(images)
    elif args.text_weight is None:
        args.text_weight = (1.0 - sum(args.img_weights)) / len(images)

    assert len(args.img_weights) == len(images)
    assert sum(args.img_weights) + args.text_weight == 1.0

    for i in range(args.n_samples):
        t1 = time.time()

        images = iter(
            model(
                args.prompt,
                images,
                args.text_weight,
                args.img_weights,
                progressive_mode="final"
            )
        ).__next__()
        # NCHW, [0, 1], float32 -> NHWC, [0, 255], uint8
        images = (
            torch.permute(images * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy()
        )

        t2 = time.time()
        execution_time = t2 - t1
        logging.info(f"Iteration {i} -- {execution_time:.6f}secs")

        # Select the first one
        image = Image.fromarray(images[0])
        image_name = "_".join(args.prompt.split(" "))
        image.save(f"{save_dir}/{image_name}_text_plus_images_{i:02d}.jpg")
