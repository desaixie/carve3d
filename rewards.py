import torchmetrics
import torch
import math
from torchvision import transforms



def compute_bounding_box(images):
    """
    images: NCHW
    returns: N=4
    """
    threshold = 0.95
    non_whitebg_masks = (images < threshold).any(dim=1)  # NHW

    bounding_boxes = []
    for mask in non_whitebg_masks:
        non_zero_indices = torch.nonzero(mask)  # N, 2 indices of non-zero (foreground) pixels

        if non_zero_indices.shape[0] == 0:
            bounding_boxes.append(torch.tensor([0, 0, 0, 0]))
        else:
            min_x = torch.min(non_zero_indices[:, 1])
            max_x = torch.max(non_zero_indices[:, 1])
            min_y = torch.min(non_zero_indices[:, 0])
            max_y = torch.max(non_zero_indices[:, 0])

            bounding_box = torch.tensor([min_x, min_y, max_x, max_y])
            bounding_boxes.append(bounding_box)

    return torch.stack(bounding_boxes).to(images.device)


def make_square_bbox(original_bbox):
    x_min, y_min, x_max, y_max = original_bbox
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    if width > height:
        delta_bot = delta_top = (width - height) // 2
        if (width - height) % 2 == 1:
            delta_top += 1
        square_bbox = [x_min, y_min - delta_top, x_max, y_max + delta_bot]
    else:
        delta_left = delta_right = (height - width) // 2
        if (height - width) % 2 == 1:
            delta_left += 1
        square_bbox = [x_min - delta_left, y_min, x_max + delta_right, y_max]

    return square_bbox



def split_tensor(tensor, tile_size=256, offset=256):
    tiles = []
    h, w = tensor.size(1), tensor.size(2)
    for y in range(int(math.ceil(h/offset))):
        for x in range(int(math.ceil(w/offset))):
            tiles.append(tensor[:, offset*y:min(offset*y+tile_size, h), offset*x:min(offset*x+tile_size, w)])
    return tiles

# TODO
def nerf_render(latents, c2ws, fxfycxcy):
    return


def gs_render(gaussians, c2ws, fxfycxcy):
    return


# TODO need a lrm_model to compute MRC
# MRC reward
class MultiviewReconstructionConsistencyReward:
    def __init__(self, config, dataset, device, project_dir, lrm_model=None, c2ws=None, fxfycxcy=None):
        self.config = config
        self.dataset = dataset
        self.device = device
        self.project_dir = project_dir
        self.lrm_model = lrm_model
        # TODO predefined camera poses, e.g. elevation=20, azimuth=[0, 90, 180, 270] in Instant3D
        self.c2ws = c2ws
        self.fxfycxcy = fxfycxcy
        self.lpips_alex = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    def compute_mrc_reward(self, mv_images):
        """mv_images: (B, 3, H, W), 4 images tiled in 2 by 2 grid"""
        batch_size = len(mv_images)
        # prepare inputs to the lrm
        input_images = []
        for i in range(batch_size):
            cell_side = mv_images[0].shape[-1] // 2
            # split into (4, 3, H/2, W/2)
            input_images.append(torch.stack(split_tensor(mv_images[i], tile_size=cell_side, offset=cell_side)))
        input_images = torch.stack(input_images)  # (B, 4, 3, H/2, W/2)
        input_c2ws = torch.stack(self.c2ws).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, 4, 4, 4)
        input_fxfycxcy = torch.stack(self.fxfycxcy).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 4, 4)
        batch = {
            "image": input_images,
            "c2ws": input_c2ws,
            "fxfycxcy": input_fxfycxcy,
        }

        self.lrm_model.eval()
        all_rewards = []
        with torch.no_grad():
            # lrm forward, get nerf/gaussians
            latents = self.lrm_model(batch)
            # gaussians = self.lrm_model(batch)

        for b in range(batch_size):
            # render
            output_images = nerf_render(latents, batch["c2ws"], batch["fxfycxcy"])  # (4, 3, H/2, W/2)
            # output_images = gs_render(gaussians, batch["c2ws"], batch["fxfycxcy"])  # (4, 3, H/2, W/2)

            bboxes = list(compute_bounding_box(input_images[b]))  # 4, 4
            rewards = []
            resize_to_render_size = transforms.Resize((output_images[0].shape[-2:]), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            for i_b, bbox in enumerate(bboxes):
                # compute lpips inside the square, padded, resized bbox
                square_bbox = make_square_bbox(bbox)
                # pad 5 pixels
                x_min, y_min, x_max, y_max = square_bbox
                x_min, y_min, x_max, y_max = x_min - 5, y_min - 5, x_max + 5, y_max + 5
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(x_max, output_images[0].shape[-1] - 1), min(y_max,output_images[0].shape[-1] - 1)
                input_bbox = resize_to_render_size(input_images[b, i_b, :, y_min:y_max + 1, x_min:x_max + 1].clone()).clamp(0.,1.).unsqueeze(0)
                output_bbox = resize_to_render_size(output_images[i_b, :, y_min:y_max + 1, x_min:x_max + 1].clone()).clamp(0.,1.).unsqueeze(0)
                reward = -self.lpips_alex(input_bbox, output_bbox)
                rewards.append(reward)
            reward = torch.stack(rewards).mean()  # 4 -> 1
            all_rewards.append(reward)

        all_rewards = torch.stack(all_rewards)  # (batch_size,)
        return all_rewards

def mrc_reward_fn(mrc, mv_images):
    return mrc.compute_mrc_reward(mv_images)