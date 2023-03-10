# 14, 15
'''
# train 실행 코드(변형코드)
config = './real_data/person_1/person_1_config.yml'
load_checkpoint = ''
train(config, load_checkpoint, debug=True)
'''
import numpy as np
import torchvision
import yaml
from easydict import EasyDict as edict
import sys
import os
import torch
import time
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter
from models.nerf.volume_rendering_utils import volume_render_radiance_field
from helper.nerf_helpers import get_minibatches, ndc_rays,sample_pdf

# base_dataset.py에서 load_flame_data 사용을 위한 import
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from datasets import base_dataset
from datasets.load_flame import load_flame_data

# nerf_helpers.py에서 get_embedding_function 사용을 위한 import
from helper import nerf_helpers

# NerFACEModel
from models import nerface_model

#import train_part
from loss import valuation_metrics


# models/nerf/train_utils.py가 원본임
def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, expressions = None, latent_code = None):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    if expressions is None:
        preds = [network_fn(batch) for batch in batches]
    elif latent_code is not None:
        preds = [network_fn(batch, expressions, latent_code) for batch in batches]
    else:
        preds = [network_fn(batch, expressions) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )

    del embedded, input_dirs_flat
    return radiance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior = None,
    latent_code = None,
    ray_dirs_fake = None
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6].clone() # TODO remove clone ablation rays
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    # Uncomment to dump a ply file visualizing camera rays and sampling points
    #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy())
    if ray_dirs_fake:
        ray_batch[...,3:6] = ray_dirs_fake[0][...,3:6]

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
        expressions,
        latent_code
    )
    # make last RGB values of each ray, the background
    if background_prior is not None:
        radiance_field[:,-1,:3] = background_prior

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        background_prior=background_prior
    )

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            expressions,
            latent_code
        )
        # make last RGB values of each ray, the background
        if background_prior is not None:
            radiance_field[:, -1, :3] = background_prior

        # Uncomment to dump a ply file visualizing camera rays and sampling points
        #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), radiance_field)

        #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), torch.softmax(radiance_field[:,:,-1],1).detach().cpu().numpy())

        #rgb_fine, disp_fine, acc_fine, _, depth_fine = volume_render_radiance_field(
        rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_render_radiance_field( # added use of weights
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            background_prior=background_prior
        )

    #return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine #added depth fine
    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,-1] #changed last return val to fine_weights


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior=None,
    latent_code = None,
    ray_directions_ablation = None
):
    is_rad = torch.is_tensor(ray_directions_ablation)
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # to return fine depth map
    if options.dataset.no_ndc is False:
        #print("calling ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        #print("calling ndc")
        #"caling normal rays (not NDC)"
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        if is_rad:
            rd_ablations = ray_directions_ablation.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if is_rad:
        rays_ablation = torch.cat((ro, rd_ablations, near, far), dim=-1)
    if options.nerf.use_viewdirs: # TODO uncomment
        rays = torch.cat((rays, viewdirs), dim=-1)
    #
    #viewdirs = None  # TODO remove this paragraph
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        if is_rad:
            viewdirs = ray_directions_ablation
            viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
            viewdirs = viewdirs.view((-1, 3))


    if is_rad:
        batches_ablation = get_minibatches(rays_ablation, chunksize=getattr(options.nerf, mode).chunksize)
    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    assert(batches[0].shape == batches[0].shape)
    background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize) if\
        background_prior is not None else background_prior
    #print("predicting")
    if is_rad:
        pred = [
            predict_and_render_radiance(
                batch,
                model_coarse,
                model_fine,
                options,
                mode,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expressions,
                background_prior = background_prior[i] if background_prior is not None else background_prior,
                latent_code = latent_code,
                ray_dirs_fake = batches_ablation
            )
            for i,batch in enumerate(batches)
        ]
    else:
        pred = [
            predict_and_render_radiance(
                batch,
                model_coarse,
                model_fine,
                options,
                mode,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expressions,
                background_prior = background_prior[i] if background_prior is not None else background_prior,
                latent_code = latent_code,
                ray_dirs_fake = None
            )
            for i,batch in enumerate(batches)
        ]
    #print("predicted")

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)

def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

def train(config, load_checkpoint, debug=False):
    # Read config file.

    cfg = None
    with open(config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = edict(cfg_dict)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None

    # Load dataset
    images, poses, render_poses, hwf, expressions = None, None, None, None, None
    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split, expressions, _, bboxs = load_flame_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            debug=debug
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    print("done loading data")
    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda" #+ ":" + str(cfg.experiment.device)
    else:
        device = "cpu"

    encode_position_fn = nerf_helpers.get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = nerf_helpers.get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = nerface_model.NerFACEModel(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        include_expression=True
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = nerface_model.NerFACEModel(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers = cfg.models.coarse.num_layers,
            hidden_size =cfg.models.coarse.hidden_size,
            include_expression=True
        )
        model_fine.to(device)

    ###################################
    ###################################
    train_background = False
    supervised_train_background = False
    blur_background = False

    train_latent_codes = True
    disable_expressions = False # True to disable expressions
    disable_latent_codes = False # True to disable latent codes
    fixed_background = True # Do False to disable BG
    regularize_latent_codes = True # True to add latent code LOSS, false for most experiments
    ###################################
    ###################################

    supervised_train_background = train_background and supervised_train_background

    if fixed_background: # load GT background
        print("loading GT background to condition on")
        from PIL import Image
        background = Image.open(os.path.join(cfg.dataset.basedir,'bg','00050.png'))
        background.thumbnail((H,W))
        background = torch.from_numpy(np.array(background).astype(np.float32)).to(device)
        background = background/255
        print("bg shape", background.shape)
        print("should be ", images[i_train][0].shape)
        assert background.shape == images[i_train][0].shape
    else:
        background = None

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    if train_background:
        #background.requires_grad = True
        #trainable_parameters.append(background) # add it later when init optimizer for different lr
        print("background.is_leaf " ,background.is_leaf, background.device)

    if train_latent_codes:
        latent_codes = torch.zeros(len(i_train),32, device=device)
        print("initialized latent codes with shape %d X %d" % (latent_codes.shape[0], latent_codes.shape[1]))
        if not disable_latent_codes:
            trainable_parameters.append(latent_codes)
            latent_codes.requires_grad = True

    if train_background:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [{'params':trainable_parameters},
             {'params':background, 'lr':cfg.optimizer.lr}],
            lr=cfg.optimizer.lr
        )
    else:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [{'params':trainable_parameters},
             {'params': background, 'lr': cfg.optimizer.lr}        ], # this is obsolete but need for continuing training
            lr=cfg.optimizer.lr
        )
    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if checkpoint["background"] is not None:
            print("loaded bg from checkpoint")
            background = torch.nn.Parameter(checkpoint['background'].to(device))
        if checkpoint["latent_codes"] is not None:
            print("loaded latent codes from checkpoint")
            latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(device))

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # Prepare importance sampling maps
    ray_importance_sampling_maps = []
    p = 0.9
    print("computing boundix boxes probability maps")
    for i in i_train:
        bbox = bboxs[i]
        probs = np.zeros((H,W))
        probs.fill(1-p)
        probs[bbox[0]:bbox[1],bbox[2]:bbox[3]] = p
        probs = (1/probs.sum()) * probs
        ray_importance_sampling_maps.append(probs.reshape(-1))


    print("Starting loop")
    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        if model_fine:
            model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None

        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        if not disable_expressions:
            expression_target = expressions[img_idx].to(device) # vector
        else: # zero expr
            expression_target = torch.zeros(76, device=device)
        #bbox = bboxs[img_idx]
        if not disable_latent_codes:
            latent_code = latent_codes[img_idx].to(device) if train_latent_codes else None
        else:
            latent_codes = torch.zeros(32, device=device)
        #latent_code = torch.zeros(32).to(device)
        ray_origins, ray_directions = nerf_helpers.get_ray_bundle(H, W, focal, pose_target)
        coords = torch.stack(
            nerf_helpers.meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )

        coords = coords.reshape((-1, 2))

        # Use importance sampling to sample mainly in the bbox with prob p
        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, p=ray_importance_sampling_maps[img_idx]
        )

        select_inds = coords[select_inds]
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        #dump_rays(ray_origins, ray_directions)

        # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
        target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
        background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (train_background or fixed_background) else None

        then = time.time()
        rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
            H,
            W,
            focal,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            expressions = expression_target,
            background_prior=background_ray_values,
            latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device)

        )
        target_ray_values = target_s

        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        loss = 0.0

        latent_code_loss = torch.zeros(1, device=device)
        if train_latent_codes and not disable_latent_codes:
            latent_code_loss = torch.norm(latent_code) * 0.0005

        background_loss = torch.zeros(1, device=device)
        if supervised_train_background:
            background_loss = torch.nn.functional.mse_loss(
                background_ray_values[..., :3], target_ray_values[..., :3], reduction='none'
            ).sum(1)
            background_loss = torch.mean(background_loss*weights) * 0.001

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        psnr = valuation_metrics.mse2psnr(loss.item())

        loss = loss + (latent_code_loss*10 if regularize_latent_codes else 0.0)
        loss_total = loss + (background_loss if supervised_train_background is not None else 0.0)
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " BG Loss: "
                + str(background_loss.item())
                + " PSNR: "
                + str(psnr)
                + " LatentReg: "
                + str(latent_code_loss.item())
            )
        #writer.add_scalar("train/loss", loss.item(), i)
        if train_latent_codes:
            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)
        if supervised_train_background:
            writer.add_scalar("train/bg_loss", background_loss.item(), i)

        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1 and False
        ):
            #torch.cuda.empty_cache()
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                
                # Do all validation set...
                loss = 0
                for img_idx in i_val[:2]:
                    img_target = images[img_idx].to(device)
                    #tqdm.set_description('val im %d' % img_idx)
                    #tqdm.refresh()  # to show immediately the update

                        # # save val image for debug ### DEBUG ####
                    # #GT = target_ray_values[..., :3]
                    # import PIL.Image
                    # #img = GT.permute(2, 0, 1)
                    # # Conver to PIL Image and then np.array (output shape: (H, W, 3))
                    # #im_numpy = img_target.detach().cpu().numpy()
                    # #im_numpy = np.array(torchvision.transforms.ToPILImage()(img_target.detach().cpu()))
                    #
                    # #                   im = PIL.Image.fromarray(im_numpy)
                    # im = img_target
                    # im = im.permute(2, 0, 1)
                    # img = np.array(torchvision.transforms.ToPILImage()(im.detach().cpu()))
                    # im = PIL.Image.fromarray(img)
                    # im.save('val_im_target_debug.png')
                    # ### DEBUG #### END

                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = nerf_helpers.get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ ,weights= run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        expressions = expression_target,
                        background_prior = background.view(-1,3) if (train_background or fixed_background) else None,
                        latent_code = torch.zeros(32).to(device) if train_latent_codes or disable_latent_codes else None,

                    )
                    #print("did one val")
                    target_ray_values = img_target
                    coarse_loss = valuation_metrics.img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                    curr_loss, curr_fine_loss = 0.0, 0.0
                    if rgb_fine is not None:
                        curr_fine_loss = valuation_metrics.img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                        curr_loss = curr_fine_loss
                    else:
                        curr_loss = coarse_loss
                    loss += curr_loss + curr_fine_loss

                loss /= len(i_val)
                psnr = valuation_metrics.mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)

                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                if train_background or fixed_background:
                    writer.add_image(
                        "validation/background", cast_to_image(background[..., :3]), i
                    )
                    writer.add_image(
                        "validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW'
                    )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )


        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
                "background": None
                if not (train_background or fixed_background)
                else background.data,
                "latent_codes": None if not train_latent_codes else latent_codes.data
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")
