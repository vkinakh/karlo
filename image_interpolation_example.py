import os
import argparse
import logging
import time
from datetime import datetime
from PIL import Image

import torch

from karlo.sampler.image_interpolation import ImageInterpolationSampler
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
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=False,
        help="If true, use bf16 for inference."
    )
    parser.add_argument("--weights", nargs="+", type=float, required=False,
                        help="Weights for each image. If not provided, all images will have the same weight. "
                             "Sum of weights must be 1.0.")
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

    model = ImageInterpolationSampler.from_pretrained(
        root_dir=args.root_dir,
        clip_model_path="ViT-L-14.pt",
        clip_stat_path="ViT-L-14_stats.th",
        sampling_type=args.sampling_type,
        use_bf16=args.use_bf16,
    )

    images_input = [Image.open(img_path) for img_path in args.img_paths]
    if args.weights is not None:
        assert len(args.weights) == len(images_input)
        assert sum(args.weights) == 1.0
    else:
        args.weights = [1.0 / len(images_input) for _ in range(len(images_input))]
    for i in range(args.n_samples):
        t1 = time.time()

        images = iter(
            model(
                images_input,
                weights=args.weights,
                progressive_mode="final",
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
        image_name = "_".join(args.img_paths[0].split("/")[-1].split(".")[:-1])
        image.save(f"{save_dir}/{image_name}_interpolate_{i:02d}.jpg")
