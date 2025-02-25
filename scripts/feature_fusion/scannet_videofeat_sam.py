import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from fusion_util import PointCloudToImageMapper
from scipy.spatial import ConvexHull
from transformers import AutoImageProcessor, AutoModel
from collections import defaultdict
import time
from plyfile import PlyData

#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from torch.multiprocessing import Pool, set_start_method

import requests
import base64
from PIL import Image
import io
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)



# Initialize the mobile_SAM model once
model_type = "vit_t"
sam_checkpoint = "./checkpoints/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# Create the mask generator once
mask_generator = SamAutomaticMaskGenerator(
    model=mobile_sam,
    points_per_side=16,  # Adjust as needed
    pred_iou_thresh=0.95,  # Filter out low-quality masks
    stability_score_thresh=0.95,  # Filter based on mask stability
    min_mask_region_area=500,  # Remove small masks
    box_nms_thresh=0.7,  # Non-maximum suppression threshold for boxes
    crop_nms_thresh=0.7  # Non-maximum suppression threshold for crops
)

needed_categories = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 
                            'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 
                                    'toilet', 'sink', 'bathtub', 'other furniture']


# # Initialize the SAM model once
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_type = "vit_t"  # Options: "vit_h", "vit_l", "vit_b"
# checkpoint_path = "./checkpoints/mobile_sam.pth"  # Update with your checkpoint path

# sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
# sam.to(device=device)

# # Create the mask generator once
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=16,  # Adjust as needed
#     pred_iou_thresh=0.95,  # Filter out low-quality masks
#     stability_score_thresh=0.95,  # Filter based on mask stability
#     min_mask_region_area=500,  # Remove small masks
#     box_nms_thresh=0.7,  # Non-maximum suppression threshold for boxes
#     crop_nms_thresh=0.7  # Non-maximum suppression threshold for crops
# )


def get_img_embed(image_paths):
    # st_time = time.time()
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        images.append(image)
    hidden_states = []
    bz = 16
    for i in range(0, len(images), bz):
        inputs = processor(images=images[i:i+bz], return_tensors="pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        hidden_states.append(last_hidden_states[:, 1:].detach().cpu().reshape(-1, 16, 16, 1024))
    # print(time.time() - st_time)
    return torch.cat(hidden_states, 0)

def save_seg_masks(image_paths, output_dir):
    """
    Perform instance segmentation on a list of images using SAM and save the masks to tensor files and mask images.

    Args:
        image_paths (list): List of image file paths.
        output_dir (str): Directory to save the output tensor files and mask images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in tqdm(image_paths):
        #output_tensor_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_masks.pt'))
        output_image_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_mask_image.jpg'))
        if os.path.exists(output_image_path):
            continue
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Ensure the image is in RGB format
        if image.ndim == 2 or image.shape[2] == 1:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Generate masks for the image
        masks = mask_generator.generate(image)

        # # Save masks to tensor file
        # torch.save(masks, output_tensor_path)

        # Create a mask image with different colors for each mask
        mask_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for i, mask in enumerate(masks):
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            mask_image[mask['segmentation']] = color

        # Save the mask image
        Image.fromarray(mask_image).save(output_image_path)


def read_seg_masks(mask_image_path):
    """
    Read the mask image and recover the masks.

    Args:
        mask_image_path (str): Path to the mask image.

    Returns:
        list: A list of masks recovered from the mask image.
    """
    mask_image = np.array(Image.open(mask_image_path))
    unique_colors = np.unique(mask_image.reshape(-1, mask_image.shape[2]), axis=0)

    recovered_masks = []
    for color in unique_colors:
        if np.all(color == [0, 0, 0]):  # Skip the background color
            continue
        recovered_mask = (mask_image == color).all(axis=-1)
        recovered_masks.append(recovered_mask)

    return recovered_masks


# def save_seg_masks(image_paths, output_dir):
#     """
#     Perform instance segmentation on a list of images using SAM and save the masks to tensor files.

#     Args:
#         image_paths (list): List of image file paths.
#         output_dir (str): Directory to save the output tensor files.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for image_path in tqdm(image_paths):
#         output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_masks.pt'))
#         if os.path.exists(output_path):
#             continue
#         image = Image.open(image_path).convert('RGB')
#         image = np.array(image)

#         # Ensure the image is in RGB format
#         if image.ndim == 2 or image.shape[2] == 1:
#             image = np.stack([image] * 3, axis=-1)
#         elif image.shape[2] == 4:
#             image = image[:, :, :3]

#         # Generate masks for the image
#         masks = mask_generator.generate(image)

#         # Save masks to tensor file
#         #masks_tensor = torch.tensor([mask['segmentation'] for mask in masks], dtype=torch.bool)
        
#         torch.save(masks, output_path)  

def seg_by_SAM(image_paths):
    """
    Perform instance segmentation on a list of images using SAM, filtering out overlapping or duplicate masks.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        list: A list of lists containing segmentation masks for each image.
    """
    masks_list = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Ensure the image is in RGB format
        if image.ndim == 2 or image.shape[2] == 1:
            # Convert grayscale to RGB by repeating the channels
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            # Convert RGBA to RGB by discarding the alpha channel
            image = image[:, :, :3]

        # Generate masks for the image
        masks = mask_generator.generate(image)
        masks_list.append(masks)
    #print(f"Segmented {len(image_paths)} images")

    return masks_list


# def seg_by_SAM_online(image_paths, api_key):
#     """
#     Perform instance segmentation on a list of images using Segmind's SAM API.

#     Args:
#         image_paths (list): List of image file paths.
#         api_key (str): Your Segmind API key.

#     Returns:
#         list: A list containing segmentation results for each image.
#     """
#     url = "https://api.segmind.com/v1/sam-img2img"
#     headers = {'x-api-key': api_key}
#     results = []

#     for image_path in image_paths:
#         # Open and convert image to base64
#         with open(image_path, "rb") as image_file:
#             image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

#         # Prepare the payload
#         data = {
#             "image": image_base64,
#             "base64": True,
#             "overlay_mask": True
#         }

#         # Send the request to the API
#         response = requests.post(url, json=data, headers=headers)

#         if response.status_code == 200:
#             response_data = response.json()
#             masks_info = response_data.get('masks', [])
#             # Decode the base64 response to an image
#             result_image_base64 = response_data.get('image')
#             result_image_data = base64.b64decode(result_image_base64)
#             result_image = Image.open(io.BytesIO(result_image_data))
#             results.append((result_image, masks_info))
#         else:
#             print(f"Error {response.status_code}: {response.text}")
#             results.append(None)

#     return results


def visualize_projection(b, shot_mask, mask_bbox, image, output_dir="output_images"):
    """
    Visualize the projection of 3D points onto a 2D image and save the output images.
    Args:
        b (int): Batch index.
        shot_mask (torch.Tensor): Tensor of shape (H, W) containing the mask.
        mask_bbox (tuple): Bounding box of the mask in format (x0, y0, x1, y1).
        image (torch.Tensor): Tensor of shape (3, H, W) containing the image.
        output_dir (str): Directory to save the output images.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    mask = shot_mask
    plt.imshow(mask, alpha=0.5, cmap='Reds')

    # Add bounding box
    x0, y0, x1, y1 = mask_bbox
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(rect)

    plt.title(f"Projection on Image {b}")

    output_path = os.path.join(output_dir, f"projection_{b}.jpg")
    plt.savefig(output_path)
    plt.close()


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='')
    parser.add_argument('--data_mode', type=str, default='mask3d', help='GT / mask3d')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args

def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''
    # if not os.path.exists(data_path):
    #     return
    scene_id = data_path.split('/')[-2]
    out_path = os.path.join(out_dir, f"{scene_id}.pt")
    if os.path.exists(out_path):
        return
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale

    # load 3D data (point cloud)
    if args.data_mode == "mask3d":
        locs_in = np.load(data_path)[:, :3]
        ins_path = f"/mnt/ssd/liuchao/Chat-Scene/data/mask3d_ins_data/pcd_all/{scene_id}.pth"
        if not os.path.exists(ins_path):
            return
        _, _, instance_class_labels, inst_segids = torch.load(f"/mnt/ssd/liuchao/Chat-Scene/data/mask3d_ins_data/pcd_all/{scene_id}.pth")
    elif args.data_mode == "odin":
        locs_in, _, _, inst_segids = torch.load(f"/mnt/ssd/liuchao/odin/odin_3d_ins_seg/pcd_all/{scene_id}.pth")
    elif args.data_mode == "GT":
        pc_infos = np.load(data_path)
        locs_in = pc_infos[:, :3]
        inst_segment_id = pc_infos[:, -1].astype(int)
        inst_num = inst_segment_id.max() + 1
        tmp_range = np.arange(inst_segment_id.shape[0])
        inst_segids = []
        for inst_id in range(inst_num):
            inst_segids.append(tmp_range[inst_segment_id == inst_id].tolist())
    elif args.data_mode == "test":
        plydata = PlyData.read(open(data_path, 'rb'))
        points = np.array([list(x) for x in plydata.elements[0]])
        locs_in = np.ascontiguousarray(points[:, :3])
        inst_segids = torch.load(f"data/mask3d_ins_data_test/pcd_all/{scene_id}.pth")[3]
    
    n_points = locs_in.shape[0]

    scene = os.path.join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(os.path.join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)

    n_points_cur = n_points
    
    inst_num = len(inst_segids)
    # volume = torch.zeros((inst_num, num_img), dtype=float, device=device)
    # crop_bbox = torch.zeros((inst_num, num_img, 4), dtype=float, device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    inst_img_feats = defaultdict(list)
    
    img_dinov2_feats = get_img_embed(img_dirs) # (num_img, 16, 16, 1024)
    
    #reverse the order of img_dirs
    #save_seg_masks( img_dirs, os.path.join(out_dir, f"{scene_id}") )
    
    # start_time = time.time()
    all_img_seg_masks = seg_by_SAM(img_dirs)
    # end_time = time.time()
    # print(f"Time taken to segment images: {end_time - start_time:.2f} seconds")

    for img_id, img_dir in enumerate(img_dirs):
        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # #load image and convert to tensor
        # image = imageio.v2.imread(img_dir)
        per_img_seg_masks = all_img_seg_masks[img_id]
        
        #use SAM to seg image
        
        
        # # Record the duration time of read_seg_masks
        # start_time = time.time()
        # mask_image_path = os.path.join(seg_masks_path, os.path.basename(img_dir).replace('.jpg', '_mask_image.jpg'))
        # seg_masks = read_seg_masks(mask_image_path)
        # end_time = time.time()
        # print(f"Time taken to read segmentation masks: {end_time - start_time:.2f} seconds")
        # mask_image_path = os.path.join(seg_masks_path, os.path.basename(img_dir).replace('.jpg', '_mask_image.jpg'))
        # seg_masks = read_seg_masks(mask_image_path)
        # print( seg_masks[0].shape )
        
        seg_masks_used = torch.zeros((len(per_img_seg_masks)), dtype=bool)
        
        # for idx, seg_mask in  enumerate(seg_masks):
        #     visualize_projection(idx, seg_mask['segmentation'], image, output_dir=f"{out_dir}/{scene_id}/{img_id}")
        
        # continue
        
        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask

        mask_ids = torch.arange(mask.shape[0])[mask.bool()].tolist()

        H, W = depth.shape
        delta_H, delta_W = H // 16, W // 16

        # compute convex hull
        for instid in range(inst_num):
            inst_seg = inst_segids[instid]
            overlap_ids = list(set(mask_ids).intersection(set(inst_seg)))
            single_inst_points = mapping[overlap_ids][:, 1:3]
            if len(single_inst_points) < 3: continue
            #points_num = single_inst_points.shape[0]
            
            max_overlap = 0
            best_match_idx = None
            for idx, seg_mask in enumerate(per_img_seg_masks):
                if seg_masks_used[idx]: continue
                mask = seg_mask['segmentation']
                #mask_area = seg_mask['area']
                overlap_ratio = np.sum(mask[single_inst_points[:, 0], single_inst_points[:, 1]])
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    best_match_idx = idx
            if best_match_idx is not None:
                seg_masks_used[best_match_idx] = True
                x0, y0, w, h = per_img_seg_masks[best_match_idx]['bbox']
                x1, y1 = x0 + w, y0 + h 
                # print(  x0, y0, x1, y1 )
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                area = per_img_seg_masks[best_match_idx]['area']
                # if any(category in instance_class_labels[instid] for category in needed_categories):
                #     visualize_projection(f"{img_id}", per_img_seg_masks[best_match_idx]['segmentation'], (x0, y0, x1, y1), image, output_dir=f"./test_sam_results/{instance_class_labels[instid]}/{scene_id}_{instid}")
                
                crop_img_feats = img_dinov2_feats[img_id, (y0 // delta_H):((y1+delta_H-1) // delta_H), (x0 // delta_W):((x1+delta_W-1) // delta_W)]
                inst_img_feats[instid].append((area, crop_img_feats.flatten(0, 1).mean(0).cpu()))
                norm_ = crop_img_feats.flatten(0, 1).mean(0).cpu().norm()
                # print(f"norm: {norm_}")

    all_feats = {}
    for instid in range(inst_num):
        if instid not in inst_img_feats:
            continue
        inst_tmp = inst_img_feats[instid]
        multivies_inst_img_feats = []
        tot_weight = sum([p[0] for p in inst_tmp])
        for weight, feat in inst_tmp:
            multivies_inst_img_feats.append({"weight": weight / tot_weight, "feat": feat.detach()})
        multivies_inst_img_feats.sort(key=lambda x: x["weight"], reverse=True)
        all_feats[f"{scene_id}_{instid:02}"] = multivies_inst_img_feats[:4] if len(multivies_inst_img_feats) > 4 else multivies_inst_img_feats
    print(f"{scene_id}: {len(all_feats)}/{inst_num}")
    torch.save(all_feats, out_path)

    # all_feats = {}
    # for instid in range(inst_num):
    #     if instid not in inst_img_feats:
    #         continue
    #     inst_tmp = inst_img_feats[instid]
    #     inst_img_feat = torch.zeros(1024, dtype=torch.float32)
    #     tot_weight = sum([p[0] for p in inst_tmp]).cpu()
    #     for weight, feat in inst_tmp:
    #         inst_img_feat += (weight / tot_weight) * feat.cpu()
    #     all_feats[f"{scene_id}_{instid:02}"] = inst_img_feat.detach()
    # print(f"{scene_id}: {len(all_feats)}/{inst_num}")
    # torch.save(all_feats, out_path)


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    # img_dim = (320, 240)
    img_dim = (640, 480)
    depth_scale = 1000.0
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary

    data_dir = args.data_dir

    data_root_2d = os.path.join(data_dir, 'scannet_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # load intrinsic parameter
    intrinsics=np.loadtxt(os.path.join(args.data_root_2d, 'intrinsics.txt'))

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsics,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)
    
    if args.data_mode == "mask3d" or args.data_mode == "GT":
        data_paths = sorted(glob(os.path.join(data_dir, 'scans/*/pc_infos.npy')))  # processed scannet data infos, can be downloaded from https://huggingface.co/datasets/ZzZZCHS/processed_scannet/blob/main/scans.tar.gz
    elif args.data_mode == "test":
        data_paths = sorted(glob(os.path.join(data_dir, 'scans_test/*/*_vh_clean_2.ply'))) # for test split

    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass

    # num_gpus = torch.cuda.device_count()
    # pool = Pool(num_gpus)

    # def process_scene_wrapper(data_path):
    #     gpu_id = torch.cuda.current_device()
    #     torch.cuda.set_device(gpu_id)
    #     process_one_scene(data_path, out_dir, args)

    # pool.map(process_scene_wrapper, data_paths)
    # pool.close()
    # pool.join()
    #data_paths = data_paths[::-1]
    for data_path in tqdm(data_paths):
       process_one_scene(data_path, out_dir, args)
    
    all_feats = {}
    for filename in os.listdir(out_dir):
        if filename.endswith('.pt'):
            all_feats.update(torch.load(os.path.join(out_dir, filename), map_location='cpu'))
    torch.save(all_feats, os.path.join(data_dir, "scannet_mask3d_sam_videofeats.pt"))

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)

