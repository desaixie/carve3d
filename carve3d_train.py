# Adapted from ddpo-pytorch/scripts/train.py
# based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
import os

from collections import defaultdict
import contextlib
from pathlib import Path
import glob
import re
from concurrent import futures
import time
from functools import partial
from typing import Dict
import random
from datetime import timedelta

from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import tqdm

import prompts as prompt_fns  # TODO release
import rewards as reward_fns
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from diffusers_patch.pipeline_MV_SDXL_with_logprob import MV_SDXLPipeline, \
    pipeline_mv_sdxl_with_logprob
from diffusers_patch.ddim_with_logprob import ddim_step_with_logprob


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
flags.DEFINE_integer('exp_num', None, 'exp_num', lower_bound=0)
flags.DEFINE_integer('num_epochs', None, 'num_epochs', lower_bound=0)
flags.DEFINE_string('run_name', None, 'run_name')
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    if FLAGS.num_epochs is not None:
        config.num_epochs = FLAGS.num_epochs

    if FLAGS.exp_num is None:
        # +1 of the latest exp
        regex = 'exp(\d*).*'
        dirs = glob.glob(f"{config.logdir}/exp*/")
        exp_nums = [int(re.search(regex, d).group(1)) for d in dirs]
        try:
            last_id = sorted(exp_nums)[-1]
        except:
            last_id = 0
    else:
        last_id = FLAGS.exp_num - 1
    if FLAGS.run_name is not None:
        config.run_name = FLAGS.run_name
    else:
        if config.run_name == f'exp{last_id}':
            config.run_name = f'exp{last_id + 1}'
        if not config.run_name:
            config.run_name = f"exp{last_id + 1}"
        # else:
        #     config.run_name += "_" + unique_id
    project_dir = os.path.join(config.logdir, config.run_name)

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=project_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        kwargs_handlers=[process_group_kwargs]
    )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, force_upcast=False)
    pipeline = MV_SDXLPipeline.from_pretrained(config.pretrained.model, vae=vae)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_vae_tiling()
    pipeline.enable_attention_slicing()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:  # only move unet to inference_dtype when using lora, since it is not trainable
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    unet_reference = None
    if config.kl_penalty or config.kl_log:
        unet_reference = UNet2DConditionModel.from_pretrained(config.pretrained.model, subfolder='unet')
        unet_reference.requires_grad_(False)
        unet_reference.to(accelerator.device, dtype=inference_dtype)
        unet_reference.eval()

    if config.use_lora:
        # Set correct lora layers
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
        for name, attn_processor in pipeline.unet.attn_processors.items():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
            module = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=config.lora_rank
            )
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())
        pipeline.unet.set_attn_processor(unet_lora_attn_procs)

    # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

    # set up diffusers-friendly checkpoint saving with Accelerate
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(pipeline.unet))):
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(pipeline.text_encoder))):
                    text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(pipeline.text_encoder_2))):
                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            MV_SDXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(pipeline.unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(pipeline.text_encoder))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(pipeline.text_encoder_2))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

        text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
        )

        text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet_lora_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(prompt_fns, config.prompt_fn)
    reward_fn = getattr(reward_fns, config.reward_fn)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )
    if config.kl_per_prompt_stat_tracking:
        kl_stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    pipeline.unet, optimizer = accelerator.prepare(pipeline.unet, optimizer)
    # if isinstance(pipeline.unet, DistributedDataParallel):
    # print("type" + str(type(pipeline.unet)))  # torch.nn.parallel.distributed.DistributedDataParallel
    pipeline.unet.config = pipeline.unet.module.config

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running Carve3D training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    dreamfusion_dataset = prompt_fns.DreamFusionPrompts(config, is_main_process=accelerator.is_local_main_process)
    mrc_reward_fn = reward_fns.MultiviewReconstructionConsistencyReward(config, dreamfusion_dataset, accelerator.device, project_dir)

    #################### VALIDATION ####################
    epoch = 0
    val_time = time.time()
    num_val_current = config.val_size // accelerator.num_processes  # val size for current process
    val_prompts = dreamfusion_dataset.val_set[
                  accelerator.process_index * num_val_current:(accelerator.process_index + 1) * num_val_current]  # 12
    val_ids = list(
        range(accelerator.process_index * num_val_current, (accelerator.process_index + 1) * num_val_current))
    with torch.no_grad(), autocast():
        pipeline.unet.eval()
        initial_latents = torch.load("eval_init_latents.pt", map_location=accelerator.device).repeat(num_val_current, 1, 1, 1)
        images = pipeline(
            prompt=val_prompts,
            latents=initial_latents,
            num_inference_steps=config.sample.num_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            num_images_per_prompt=1,
            output_type="pt",
        )

    # compute rewards, save nerfvis.html and tiled.png
    images = images.to(torch.float32)
    all_rewards = mrc_reward_fn.compute_mrc_reward(images)

    # gather on all processes, so that wait_for_everyone works correctly. otherwise (only do gather on main process) stuck on wait_for_everyone
    all_rewards = accelerator.gather(all_rewards).reshape(-1).cpu().numpy()

    if accelerator.is_main_process:  # main process downloads the epoch, writes htmls, and uploads htmls
        accelerator.log(
            {
                "val_reward": all_rewards,
                "val_reward_mean": all_rewards.mean(),
                "val_reward_std": all_rewards.std(),
            },
            step=global_step,
        )
        print(f"validation takes {time.time() - val_time}")

    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        if epoch == first_epoch:  # fill stat tracker
            prompts_all = [p for p in dreamfusion_dataset.train_set for _ in
                           range(config.per_prompt_stat_tracking.min_count)]
            prompt_ids_all = [id for id in range(len(dreamfusion_dataset.train_set)) for _ in
                              range(config.per_prompt_stat_tracking.min_count)]
            fill = config.sample.batch_size * config.sample.num_batches_per_epoch * config.num_nodes * 8 - len(prompts_all)
            if fill > 0:
                prompts_all.extend(prompts_all[:fill])
                prompt_ids_all.extend(prompt_ids_all[:fill])
            idxs = list(range(config.sample.batch_size * config.sample.num_batches_per_epoch * accelerator.process_index,
                              config.sample.batch_size * config.sample.num_batches_per_epoch * (accelerator.process_index + 1)))
            prompts_all = [prompts_all[idx] for idx in idxs]
            prompt_ids_all = [prompt_ids_all[idx] for idx in idxs]

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # sample prompts
            if epoch == first_epoch:
                prompts = prompts_all[i * config.sample.batch_size:(i + 1) * config.sample.batch_size]
                prompt_ids = prompt_ids_all[i * config.sample.batch_size:(i + 1) * config.sample.batch_size]
                prompt_ids = torch.tensor(prompt_ids).to(device=accelerator.device, dtype=torch.int)
            else:
                prompts, prompt_ids = prompt_fn(dreamfusion_dataset, config.sample.batch_size)
                prompt_ids = torch.tensor(prompt_ids).to(device=accelerator.device, dtype=torch.int)

            # decide whether to use kl unet reference
            if config.kl_penalty or (config.kl_log and epoch % config.val_freq == 0):
                do_kl_unet_reference = unet_reference
            else:
                do_kl_unet_reference = None

            # sample model outputs
            with autocast(), torch.no_grad():
                # TODO release
                initial_latents = None
                # denoising inference pipline
                images, latents, log_probs, log_probs_reference, prompt_embeds, negative_prompt_embeds, add_text_embeds, negative_pooled_prompt_embeds, add_time_ids = pipeline_mv_sdxl_with_logprob(
                    pipeline,
                    prompt=prompts,
                    # prompt_embeds=prompt_embeds,
                    # negative_prompt_embeds=sample_neg_prompt_embeds,
                    latents=initial_latents,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    num_images_per_prompt=1,
                    output_type="pt",
                    unet_reference=do_kl_unet_reference
                )

            # postprocess samples
            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 32, 32)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
            if do_kl_unet_reference is not None:
                log_probs_reference = torch.stack(log_probs_reference, dim=1)  # (batch_size, num_steps)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)
            single_add_time_id = torch.tensor([1024., 1024., 0., 0., 1024., 1024.], dtype=add_time_ids.dtype, device=add_time_ids.device)

            # compute rewards asynchronously
            images = images.to(torch.float32)
            rewards = mrc_reward_fn.compute_mrc_reward(images)
            time.sleep(0)

            assert torch.equal(add_time_ids, single_add_time_id.repeat(prompt_embeds.shape[0]*2, 1)), f"add_time_ids: {add_time_ids}"
            assert torch.equal(negative_prompt_embeds, torch.zeros_like(prompt_embeds)), f"negative_prompt_embeds: {negative_pooled_prompt_embeds}"
            assert torch.equal(negative_pooled_prompt_embeds, torch.zeros_like(add_text_embeds)), f"negative_pooled_prompt_embeds: {negative_pooled_prompt_embeds}"

            if do_kl_unet_reference is not None:
                kl = (log_probs - log_probs_reference).mean(dim=1)  # batch_size, . mean over timesteps
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,  # split negative and the actual prompt_embeds
                        "add_text_embeds": add_text_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "log_probs_reference": log_probs_reference,
                        "kl": kl,
                        "rewards": rewards,
                    }
                )
            else:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,  # split negative and the actual prompt_embeds
                        "add_text_embeds": add_text_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards": rewards,
                    }
                )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()  # batch_size * num_processes, 1
        # log rewards and images
        accelerator.log(
            {"reward": rewards, "epoch": epoch, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )

        # per-prompt mean/std tracking on gathered prompts and rewards
        prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
        prompts = dreamfusion_dataset.get_prompts_from_ids(prompt_ids)
        # prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        if do_kl_unet_reference is not None:
            kl = accelerator.gather(samples["kl"]).cpu().numpy()
            if accelerator.is_main_process:
                accelerator.log({"kl_sampled": kl, "kl_sampled_mean": kl.mean(), "kl_sampled_std": kl.std()}, step=global_step)
            if config.kl_penalty and config.kl_per_prompt_stat_tracking:
                kl_normalized = kl_stat_tracker.update(prompts, kl)  # normalize

        if config.per_prompt_stat_tracking:
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        if config.kl_penalty and config.kl_in_reward:
            if config.kl_per_prompt_stat_tracking:
                advantages = advantages - config.kl_normalized_coeff * kl_normalized
            else:
                advantages = advantages - config.kl_normalized_coeff * kl

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]
        # del samples["prompt_ids"]  # don't delete, so the main process won't gather on deleted prompt_ids

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps

        if epoch == first_epoch:
            # fill stat tracker before training starts
            for key, itm in stat_tracker.stats.items():
                if len(itm) < stat_tracker.min_count:
                    print(f"WARNING: prompt {key} has only {len(itm)} samples")
            continue

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            train_keys = ["timesteps", "latents", "next_latents", "log_probs"]
            if do_kl_unet_reference is not None:
                train_keys.append("log_probs_reference")
            for key in train_keys:
                samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            pipeline.unet.train()
            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: RL training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                prompt_embeds = torch.cat([torch.zeros_like(sample["prompt_embeds"]), sample["prompt_embeds"]])
                text_embeds = torch.cat([torch.zeros_like(sample["add_text_embeds"]), sample["add_text_embeds"]])
                added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": single_add_time_id.repeat(prompt_embeds.shape[0], 1)}

                for j in tqdm(
                        range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(pipeline.unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = pipeline.unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=added_cond_kwargs,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            else:
                                raise NotImplementedError
                            # compute the log prob of next_latents (x_{t-1}, more denoised) given latents (x_t) under the current model ('s noise_pred)
                            # nominator in eq. (4)
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"], -config.train.adv_clip_max, config.train.adv_clip_max
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])  # importance sampling, the fraction in eq. (4)

                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if do_kl_unet_reference is not None:
                            kl_penalty = (log_prob - sample["log_probs_reference"][:, j]).mean()
                            if config.kl_penalty and not config.kl_in_reward:
                                loss += config.kl_beta * kl_penalty

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        # kl between model after this train step and model during sampling. If high, model did a very large gradient step.
                        # Used for traditional RL (PPO) logging, not directly related to RLHF PPO
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        # fraction of the training data that triggered the clipped ratio
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["loss"].append(loss)
                        if do_kl_unet_reference is not None:
                            info["kl"].append(kl_penalty)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet_lora_parameters, config.train.max_grad_norm)
                        # below are only performed after reaches num_train_timesteps * train.gradient_accumulation_steps
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step on the accumulated gradients behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch % config.save_freq == 0 and epoch != first_epoch and accelerator.is_main_process:
        #     accelerator.save_state()  # saves those in accelerator.prepare(...)
            epoch_dir = Path(project_dir) / str(epoch)
            epoch_dir.mkdir(parents=True, exist_ok=True)
            unet_lora_layers = unet_attn_processors_state_dict(accelerator.unwrap_model(pipeline.unet))
            MV_SDXLPipeline.save_lora_weights(
                save_directory=epoch_dir,
                unet_lora_layers=unet_lora_layers,
            )

        # evaluate on validation set
        if (epoch % config.val_freq == 0 and epoch != first_epoch) or epoch == config.num_epochs - 1:
            val_time = time.time()
            num_val_current = config.val_size // accelerator.num_processes  # val size for current process
            val_prompts = dreamfusion_dataset.val_set[accelerator.process_index*num_val_current:(accelerator.process_index+1)*num_val_current]  # 12
            val_ids = list(range(accelerator.process_index*num_val_current,(accelerator.process_index+1)*num_val_current))
            with torch.no_grad():
                with autocast():
                    # TODO release
                    initial_latents = torch.load("eval_init_latents.pt", map_location=accelerator.device).repeat(num_val_current, 1, 1, 1)
                    # images = pipeline(
                    images, _, log_probs, log_probs_reference, _, _, _, _, _ = pipeline_mv_sdxl_with_logprob(
                        pipeline,
                        prompt=val_prompts,
                        # prompt_embeds=prompt_embeds,
                        # negative_prompt_embeds=sample_neg_prompt_embeds,
                        latents=initial_latents,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        num_images_per_prompt=1,
                        output_type="pt",
                        unet_reference=unet_reference
                    )
            # kl
            if config.kl_log:
                log_probs = torch.stack(log_probs)
                log_probs_reference = torch.stack(log_probs_reference)
                kl_penalty = (log_probs - log_probs_reference).mean()
                kl_penalty = accelerator.gather(kl_penalty).reshape(-1).cpu().numpy()
                if accelerator.is_main_process:
                    accelerator.log({"val_kl": kl_penalty, "val_kl_mean": kl_penalty.mean(), "val_kl_std": kl_penalty.std()}, step=global_step)

            # compute rewards, save nerfvis.html and tiled.png
            images = images.to(torch.float32)
            all_rewards = mrc_reward_fn.compute_mrc_reward(images)

            # gather on all processes, so that wait_for_everyone works correctly. otherwise (only do gather on main process) stuck on wait_for_everyone
            all_rewards = accelerator.gather(all_rewards).reshape(-1).cpu().numpy()

            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "val_reward": all_rewards,
                        "val_reward_mean": all_rewards.mean(),
                        "val_reward_std": all_rewards.std(),
                    },
                    step=global_step,
                )
                accelerator.log(
                    {
                        "images": [
                            wandb.Image(str(Path(project_dir) / str(epoch) / str(prompt_i) / f'together.png'), caption=f"{reward:.4f}")
                            for prompt_i, reward in enumerate(all_rewards)
                        ],
                    },
                    step=global_step,
                )
                print(f"validation takes {time.time() - val_time}")
            accelerator.wait_for_everyone()  # wait for main process for uploading and downloading

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(pipeline.unet)
        unet_lora_layers = unet_attn_processors_state_dict(unet)
        MV_SDXLPipeline.save_lora_weights(
            save_directory=project_dir,
            unet_lora_layers=unet_lora_layers,
        )


if __name__ == "__main__":
    app.run(main)
