from typing import Iterator, List

import torch
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode

from .template import BaseSampler, CKPT_PATH


class TextPlusImagesSampler(BaseSampler):
    """
    A sampler that generates images using images and text conditioning.

    :param root_dir: directory for model checkpoints.
    :param sampling_type: ["default", "fast"]
    :param use_bf16: If true, use bf16 for inference.
    """

    def __init__(
        self,
        root_dir: str,
        sampling_type: str = "default",
        use_bf16: bool = False,
    ):
        super().__init__(root_dir, sampling_type, use_bf16)

    @classmethod
    def from_pretrained(
        cls,
        root_dir: str,
        clip_model_path: str,
        clip_stat_path: str,
        sampling_type: str = "default",
        use_bf16: bool = False,
    ):

        model = cls(
            root_dir=root_dir,
            sampling_type=sampling_type,
            use_bf16=use_bf16,
        )
        model.load_clip(clip_model_path)
        model.load_prior(
            f"{CKPT_PATH['prior']}",
            clip_stat_path=clip_stat_path,
        )
        model.load_decoder(f"{CKPT_PATH['decoder']}")
        model.load_sr_64_256(CKPT_PATH["sr_256"])

        return model

    def preprocess_text(
        self,
        prompt: str,
    ):
        """Setup prompts & cfg scales"""
        prompts_batch = [prompt]

        prior_cf_scales_batch = [self._prior_cf_scale] * len(prompts_batch)
        prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="cuda")

        decoder_cf_scales_batch = [self._decoder_cf_scale] * len(prompts_batch)
        decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="cuda")

        """ Get CLIP text feature """
        clip_model = self._clip
        tokenizer = self._tokenizer
        max_txt_length = self._prior.model.text_ctx

        tok, mask = tokenizer.padded_tokens_and_mask(prompts_batch, max_txt_length)
        cf_token, cf_mask = tokenizer.padded_tokens_and_mask([""], max_txt_length)
        if not (cf_token.shape == tok.shape):
            cf_token = cf_token.expand(tok.shape[0], -1)
            cf_mask = cf_mask.expand(tok.shape[0], -1)

        tok = torch.cat([tok, cf_token], dim=0)
        mask = torch.cat([mask, cf_mask], dim=0)

        tok, mask = tok.to(device="cuda"), mask.to(device="cuda")
        txt_feat, txt_feat_seq = clip_model.encode_text(tok)

        return (
            prompts_batch,
            prior_cf_scales_batch,
            decoder_cf_scales_batch,
            txt_feat,
            txt_feat_seq,
            tok,
            mask,
        )

    def preprocess_images(
        self,
        images: List,
    ):
        # preprocess input image
        images = [TVF.normalize(
            TVF.to_tensor(
                TVF.resize(
                    image,
                    [224, 224],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            ),
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ) for image in images]
        image_batch = torch.stack(images).cuda()

        """ Get CLIP text and image features """
        clip_model = self._clip

        img_feats = clip_model.encode_image(image_batch)
        return img_feats

    def __call__(
        self,
        prompt: str,
        images: List,
        text_weight: float,
        img_weights: List[float],
        progressive_mode=None,
    ) -> Iterator[torch.Tensor]:
        assert progressive_mode in ("loop", "stage", "final")
        assert len(images) == len(img_weights)

        with torch.no_grad(), torch.cuda.amp.autocast():
            img_feats = self.preprocess_images(images)
            (
                prompts_batch,
                prior_cf_scales_batch,
                decoder_cf_scales_batch,
                txt_feat,
                txt_feat_seq,
                tok,
                mask,
            ) = self.preprocess_text(prompt)

            # Transform CLIP text feature into image feature
            text_to_img_feat = self._prior(
                txt_feat,
                txt_feat_seq,
                mask,
                prior_cf_scales_batch,
                timestep_respacing=self._prior_sm,
            )

            # Combine text and image features by weighted sum
            # append text_to_img_feat to img_feats
            img_feats = torch.cat([img_feats, text_to_img_feat], dim=0)
            weights = torch.tensor(img_weights + [text_weight], device="cuda")
            img_feat = torch.sum(img_feats * weights.unsqueeze(1), dim=0, keepdim=True)

            """ Generate 64x64px images """
            images_64_outputs = self._decoder(
                txt_feat,
                txt_feat_seq,
                tok,
                mask,
                img_feat,
                cf_guidance_scales=decoder_cf_scales_batch,
                timestep_respacing=self._decoder_sm,
            )

            images_64 = None
            for k, out in enumerate(images_64_outputs):
                images_64 = out
                if progressive_mode == "loop":
                    yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)
            if progressive_mode == "stage":
                yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)

            images_64 = torch.clamp(images_64, -1, 1)

            """ Upsample 64x64 to 256x256 """
            images_256 = TVF.resize(
                images_64,
                [256, 256],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            images_256_outputs = self._sr_64_256(
                images_256, timestep_respacing=self._sr_sm
            )

            for k, out in enumerate(images_256_outputs):
                images_256 = out
                if progressive_mode == "loop":
                    yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)
            if progressive_mode == "stage":
                yield torch.clamp(out * 0.5 + 0.5, 0.0, 1.0)

        yield torch.clamp(images_256 * 0.5 + 0.5, 0.0, 1.0)
