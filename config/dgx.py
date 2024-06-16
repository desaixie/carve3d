import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = ""

    config.num_epochs = 200
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    # config.prompt_fn = "imagenet_animals"
    config.prompt_fn = "image_RT"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def incompressibility():
    config = compressibility()
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    config = compressibility()
    config.num_epochs = 200
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.gradient_accumulation_steps = 4

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def prompt_image_alignment():
    config = compressibility()

    config.num_epochs = 200
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # rewards
    config.reward_fn = ""

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def carve3d_train():
    config = prompt_image_alignment()

    config.kl_log = True
    config.kl_penalty = True
    config.kl_beta = 0.001
    config.kl_in_reward = True  # otherwise, in loss
    config.kl_per_prompt_stat_tracking = True
    config.kl_normalized_coeff = 0.2

    config.bbox_lpips = True
    config.curate_low_reward_prompts = True

    # defaults
    # TODO
    # config.pretrained.model = '/path/to/instant3d_10k'
    config.pretrained.model = '/path/to/your_multiview_diffusion_model'
    config.sample.num_steps = 100 # TODO number of denoising steps that your mv diffusion model requires
    config.num_epochs = 55
    config.save_freq = 1
    # TODO
    config.prompt_fn = "dreamfusion_prompts"  # TODO release
    config.reward_fn = "mrc_reward_fn"
    # random seed for reproducibility.
    config.seed = 42

    # scale
    config.num_nodes = 6  # 6
    config.train_size = 30
    config.lora_rank = 4 # The dimension of the LoRA update matrices.
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8  # 8. with SD-XL, batch size of 8 already have A100 runs at 100%, increasing batch size doesn't use more resource to sample 8*6 faster
    config.sample.num_batches_per_epoch = 12 // config.num_nodes  # 6

    config.train.batch_size = 4  # 4
    config.train.gradient_accumulation_steps = 24 // config.num_nodes  # 12

    epoch_total_batch_size = config.sample.batch_size * config.sample.num_batches_per_epoch * config.num_nodes * 8
    epoch_avg_batch_size_per_prompt = epoch_total_batch_size // config.train_size
    config.per_prompt_stat_tracking = {
        "buffer_size": epoch_avg_batch_size_per_prompt * 3,
        "min_count": epoch_avg_batch_size_per_prompt,
    }

    return config


def get_config(name):
    return globals()[name]()
