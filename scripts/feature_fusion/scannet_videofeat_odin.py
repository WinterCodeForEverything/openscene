import os
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from fusion_util import PointCloudToImageMapper
from scipy.spatial import ConvexHull
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from collections import defaultdict
import time
from plyfile import PlyData
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)


def get_img_embed(image_paths):
    # st_time = time.time()
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        #print( image.size )
        images.append(image)
    hidden_states = []
    bz = 32
    for i in range(0, len(images), bz):
        inputs = processor(images=images[i:i+bz], return_tensors="pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        hidden_states.append(last_hidden_states[:, 1:].detach().cpu().reshape(-1, 16, 16, 1024))
    # print(time.time() - st_time)
    return torch.cat(hidden_states, 0)


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--data_mode', type=str, default='mask3d', help='GT / mask3d / odin')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args

# def dinov2_mask_visualization(image_paths, dinov2_features, save_path):

#     scaler = MinMaxScaler()

#     for i, image_path in enumerate(image_paths):
#         image = Image.open(image_path).convert('RGB')
#         feature = dinov2_features[i].cpu().numpy()

#         # Normalize the feature for visualization
#         feature = scaler.fit_transform(feature.reshape(-1, feature.shape[-1])).reshape(feature.shape)

#         # Create a figure to display the image and its feature map
#         fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#         # Display the original image
#         ax[0].imshow(image)
#         ax[0].set_title('Original Image')
#         ax[0].axis('off')

#         # Display the feature map
#         ax[1].imshow(feature.mean(axis=-1), cmap='viridis')
#         ax[1].set_title('DINOv2 Feature Map')
#         ax[1].axis('off')

#         # Save the visualization
#         plt.savefig(os.path.join(save_path, f"visualize_{i}.png"))
#         plt.close(fig)


def project(intrinsics, poses, one_object_points, image_size, obj_boxes):
    """
    Projects 3D world coordinates back to 2D image coordinates and filters out invalid points considering occlusion.

    Inputs:
        intrinsics:  3 X 3 (camera intrinsics)
        poses: 4 X 4 (camera extrinsics)
        one_object_points: N X 3 (3D world coordinates)
        image_size: tuple (H, W) representing the image dimensions
        obj_boxes: K X 6 (bounding boxes of objects in the format [x_min, y_min, z_min, x_max, y_max, z_max])

    Outputs:
        valid_pixel_coords: Tensor of shape (N_valid, 2) containing only valid 2D pixel coordinates
        valid_mask: Tensor of shape (N_valid,) indicating valid points
    """
    N, _ = one_object_points.shape
    img_H, img_W = image_size

    # Convert 3D world coordinates to homogeneous form (N, 4)
    ones = torch.ones((N, 1), device=one_object_points.device)
    world_coords_h = torch.cat([one_object_points, ones], dim=-1)  # N X 4

    # Transform world coordinates to camera coordinates (N X 4)
    cam_coords_h = torch.matmul(torch.inverse(poses), world_coords_h.T).T  # N X 4

    # Convert to non-homogeneous 3D camera coordinates
    cam_coords = cam_coords_h[..., :3] / cam_coords_h[..., 3:4]  # N X 3

    # Extract intrinsics parameters
    fx = intrinsics[0, 0]  # scalar
    fy = intrinsics[1, 1]  # scalar
    px = intrinsics[0, 2]  # scalar
    py = intrinsics[1, 2]  # scalar

    # Compute 2D pixel coordinates
    x = (cam_coords[..., 0] * fx / cam_coords[..., 2]) + px
    y = (cam_coords[..., 1] * fy / cam_coords[..., 2]) + py

    # Stack pixel coordinates
    pixel_coords = torch.stack([x, y], dim=-1)  # N X 2

    # Validity mask: check if points are in front of the camera and within image bounds
    valid = (cam_coords[..., 2] > 0) & (x >= 0) & (x < img_W) & (y >= 0) & (y < img_H)

    # Check for occlusion using 3D bounding boxes
    for box in obj_boxes:
        x_min, y_min, z_min, x_max, y_max, z_max = box
        occlusion_mask = (cam_coords[..., 0] >= x_min) & (cam_coords[..., 0] <= x_max) & \
                    (cam_coords[..., 1] >= y_min) & (cam_coords[..., 1] <= y_max) & \
                    (cam_coords[..., 2] >= z_min) & (cam_coords[..., 2] <= z_max)
        valid &= ~occlusion_mask

    return pixel_coords, valid



def visualize_projection(b, shot_mask, image, output_dir="output_images"):
    """
    Visualize the projection of 3D points onto a 2D image and save the output images.
    Args:
        b (int): Batch index.
        shot_mask (torch.Tensor): Tensor of shape (H, W) containing the mask.
        image (torch.Tensor): Tensor of shape (3, H, W) containing the image.
        output_dir (str): Directory to save the output images.
    """
    import matplotlib.pyplot as plt

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the image tensor to a numpy array and transpose to (H, W, 3)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    mask = shot_mask.cpu().numpy()
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title(f"Projection on Image {b}")

    output_path = os.path.join(output_dir, f"projection_{b}.jpg")
    plt.savefig(output_path)
    plt.close()
    
    

def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''
    scene_id = data_path.split('/')[-1]
    if os.path.exists(os.path.join(out_dir, f"{scene_id}.pt")):
        return
    #print(scene_id)
    if int(scene_id[5:9]) > 706:
        return
    print(scene_id)
    
    out_path = os.path.join(out_dir, f"{scene_id}.pt")
    if os.path.exists(out_path):
        return
    #point2img_mapper = args.point2img_mapper
    intrinsics = np.loadtxt(os.path.join(args.data_root_2d, scene_id, 'intrinsic/intrinsic_depth.txt'))
    point2img_mapper = PointCloudToImageMapper(
            image_dim=args.img_dim, intrinsics=intrinsics,
            visibility_threshold=args.visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)
    depth_scale = args.depth_scale

    # load 3D data (point cloud)
    if args.data_mode == "odin":
        if not os.path.exists(f"/mnt/ssd/liuchao/odin/odin_3d_ins_seg2/{scene_id}.pth"):
            return
        locs_in, _, _, inst_seg_masks, image_feats = torch.load(f"/mnt/ssd/liuchao/odin/odin_3d_ins_seg2/{scene_id}.pth")
        inst_num = inst_seg_masks.shape[0]
        tmp_range = np.arange(locs_in.shape[0])
        inst_segids = []
        for inst_id in range(inst_num):
            inst_segids.append(tmp_range[inst_seg_masks[inst_id]].tolist())
            #print(len(inst_segids[-1]))
        #print(inst_segids.shape )
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
    #print(inst_num)
    volume = torch.zeros((inst_num, num_img), dtype=float, device=device)
    crop_bbox = torch.zeros((inst_num, num_img, 4), dtype=float, device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    inst_img_feats = defaultdict(list)
    
    #img_dinov2_feats = get_img_embed(img_dirs) # (num_img, 16, 16, 1024)
    #img_odin_feats = torch.from_numpy(image_feats['res3']).permute(0, 2, 3, 1).to(device)
    #print("img_odin_feats:", img_odin_feats.shape)
    #images = []
    for img_id, img_dir in enumerate(img_dirs):
        #load image
        image = Image.open(img_dir).convert('RGB')
        
        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.png', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth')) / depth_scale
        #print(depth)
        #depth = depth / depth_scale
        #print(depth.shape)
        #continue

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            print(f"Skip {scene_id} {img_id}")
            continue

        mapping = torch.from_numpy(mapping)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask

        mask_ids = torch.arange(mask.shape[0])[mask.bool()].tolist()

        # img_dinov2_feats = torch.zeros((16, 16, 1024), device=device)
        H, W = depth.shape
        delta_H, delta_W = H // 32, W // 40

        crop_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        # compute convex hull
        for instid in range(inst_num):
            inst_seg = inst_segids[instid]
            #print(inst_seg)
            overlap_ids = list(set(mask_ids).intersection(set(inst_seg)))
            if len(overlap_ids) == 0: continue
            #print( overlap_ids[0] )
            single_inst_points = mapping[overlap_ids][:, 1:3]
            if len(single_inst_points) < 3: continue
            try:
                hull = ConvexHull( single_inst_points )
            except:
                continue
            volume[instid, img_id] = hull.volume
            crop_bbox[instid, img_id] = torch.as_tensor(np.concatenate([single_inst_points[hull.vertices].min(axis=0)[0], single_inst_points[hull.vertices].max(axis=0)[0]]))
            if volume[instid, img_id] < delta_H * delta_W:
                continue
            x0, y0, x1, y1 = crop_bbox[instid, img_id].to(int).tolist()
            crop_mask[x0:x1, y0:y1] = True
            continue
            crop_img_feats = img_odin_feats[img_id, (x0 // delta_H):((x1+delta_H-1) // delta_H), (y0 // delta_W):((y1+delta_W-1) // delta_W)]
            inst_img_feats[instid].append((volume[instid, img_id].cpu(), crop_img_feats.flatten(0, 1).mean(0).cpu()))

        visualize_projection(img_id, crop_mask, image, output_dir=os.path.join(out_dir, scene_id))
        if img_id > 20:
            break
    
    return
    
    all_feats = {}
    for instid in range(inst_num):
        if instid not in inst_img_feats:
            continue
        inst_tmp = inst_img_feats[instid]
        inst_img_feat = torch.zeros(256, dtype=torch.float32)
        tot_weight = sum([p[0] for p in inst_tmp]).cpu()
        for weight, feat in inst_tmp:
            inst_img_feat += (weight / tot_weight) * feat.cpu()
        all_feats[f"{scene_id}_{instid:02}"] = inst_img_feat.detach()
    print(f"{scene_id}: {len(all_feats)}/{inst_num}")
    torch.save(all_feats, out_path)


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    # img_dim = (320, 240)
    args.img_dim = (640, 480)
    depth_scale = 1000.0
    #######################################
    args.visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary

    data_dir = args.data_dir

    #data_root_2d = os.path.join(data_dir,'scannet_2d')
    args.data_root_2d = data_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # # load intrinsic parameter
    # intrinsics=np.loadtxt(os.path.join(args.data_root_2d, 'intrinsics.txt'))

    # # calculate image pixel-3D points correspondances
    # args.point2img_mapper = PointCloudToImageMapper(
    #         image_dim=img_dim, intrinsics=intrinsics,
    #         visibility_threshold=visibility_threshold,
    #         cut_bound=args.cut_num_pixel_boundary)
    
    if args.data_mode == "mask3d" or args.data_mode == "GT":
        data_paths = sorted(glob(os.path.join(data_dir, 'scans/*/pc_infos.npy')))  # processed scannet data infos, can be downloaded from https://huggingface.co/datasets/ZzZZCHS/processed_scannet/blob/main/scans.tar.gz
    elif args.data_mode == "test":
        data_paths = sorted(glob(os.path.join(data_dir, 'scans_test/*/*_vh_clean_2.ply'))) # for test split
    elif args.data_mode == "odin":
        data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] #sorted( glob( os.path.join(data_dir, '*') ) )

    for data_path in data_paths:
       process_one_scene(data_path, out_dir, args)
       #break
    
    all_feats = {}
    for filename in os.listdir(out_dir):
        if filename.endswith('.pt'):
            all_feats.update(torch.load(os.path.join(out_dir, filename), map_location='cpu'))
    torch.save(all_feats, os.path.join( "/home/liuchao/Chat-Scene/annotations/scannet_odin_videofeats.pt"))

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)

