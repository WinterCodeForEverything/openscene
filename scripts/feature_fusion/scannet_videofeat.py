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


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

needed_categories = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 
                            'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 
                                    'toilet', 'sink', 'bathtub', 'other furniture']

# {'cabinet': 'good', 
#  'bed': 'good', 
#  'chair': 'good', 
#  'sofa': 'average', 
#  'table': 'average', 
#  'door': 'good', 
#  'window': 'average', 
#  'bookshelf': 'good', 
# 'picture': 'good', 
# 'counter': 'average', 
# 'desk': 'good', 
# 'curtain': 'good', 
# 'refrigerator': 'good', 
# 'shower curtain': 'average', 
# 'toilet': 'good', 
# 'sink': 'good', 
# 'bathtub': 'good',
# }

#1 big masks are usually better than small masks



def get_img_embed(image_paths):
    # st_time = time.time()
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
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
    parser.add_argument('--data_mode', type=str, default='mask3d', help='GT / mask3d')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


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

    ## Convert the image tensor to a numpy array and transpose to (H, W, 3)
    #image_np = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    mask = shot_mask.cpu().numpy()
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title(f"Projection on Image {b}")

    output_path = os.path.join(output_dir, f"projection_{b}.jpg")
    plt.savefig(output_path)
    plt.close()
    
    
    

def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''
    scene_id = data_path.split('/')[-2]
    out_path = os.path.join(out_dir, f"{scene_id}.pt")
    if os.path.exists(out_path):
        return
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale

    # load 3D data (point cloud)
    if args.data_mode == "mask3d":
        locs_in = np.load(data_path)[:, :3]
        seg_path = f"/mnt/ssd/liuchao/Chat-Scene/data/mask3d_ins_data/pcd_all/{scene_id}.pth"
        if os.path.exists(seg_path):
            _, _, instance_class_labels, inst_segids = torch.load(seg_path)
        else:
            return
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
    #print(f"{scene_id}: {inst_num} instances")
    volume = torch.zeros((inst_num, num_img), dtype=float, device=device)
    crop_bbox = torch.zeros((inst_num, num_img, 4), dtype=float, device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    # inst_img_feats = defaultdict(list)
    
    # img_dinov2_feats = get_img_embed(img_dirs) # (num_img, 16, 16, 1024)

    for img_id, img_dir in enumerate(img_dirs):
        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        #load image and convert to tensor
        image = imageio.v2.imread(img_dir)
        
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

        # img_dinov2_feats = torch.zeros((16, 16, 1024), device=device)
        H, W = depth.shape
        delta_H, delta_W = H // 16, W // 16

        # compute convex hull
        for instid in range(inst_num):
            inst_seg = inst_segids[instid]
            overlap_ids = list(set(mask_ids).intersection(set(inst_seg)))
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
            crop_mask = torch.zeros((H, W), dtype=bool, device=device)
            crop_mask[x0:x1, y0:y1] = True
            #lower the caption of instance_class_labels
            
            #if instance_class_labels[instid] in needed_categories:
            if "sofa" in instance_class_labels[instid]:
                visualize_projection(f"{img_id}", crop_mask, image, output_dir=f"{out_dir}/sofa/{scene_id}_{instid}")
                #visualize_projection(f"{img_id}", crop_mask, image, output_dir=f"{out_dir}/{instance_class_labels[instid]}/{scene_id}_{instid}")
            else:
                print(f"{scene_id}_{img_id}: {instance_class_labels[instid]}")
            continue
        
            # crop_img_feats = img_dinov2_feats[img_id, (x0 // delta_H):((x1+delta_H-1) // delta_H), (y0 // delta_W):((y1+delta_W-1) // delta_W)]
            # inst_img_feats[instid].append((volume[instid, img_id].cpu(), crop_img_feats.flatten(0, 1).mean(0).cpu()))

    # all_feats = {}
    # for instid in range(inst_num):
    #     if instid not in inst_img_feats:
    #         continue
    #     inst_tmp = inst_img_feats[instid]
    #     multivies_inst_img_feats = []
    #     #inst_img_feat = torch.zeros(1024, dtype=torch.float32)
    #     tot_weight = sum([p[0] for p in inst_tmp]).cpu()
    #     for weight, feat in inst_tmp:
    #         multivies_inst_img_feats.append({"weight": weight / tot_weight, "feat": feat.detach()})
    #         #inst_img_feat += (weight / tot_weight) * feat.cpu()
    #     #sort inst_img_feats according to weight
    #     multivies_inst_img_feats.sort(key=lambda x: x["weight"], reverse=True)
        
    #     all_feats[f"{scene_id}_{instid:02}"] = multivies_inst_img_feats[:4] if len(multivies_inst_img_feats) > 4 else multivies_inst_img_feats
    #     #inst_img_feat.detach()
    #     #print(f"{scene_id}_{instid:02}: {len(multivies_inst_img_feats)}/{len(inst_tmp)}")
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

    #data_paths = data_paths[::-1]
    for data_path in data_paths:
       process_one_scene(data_path, out_dir, args)
    
    # all_feats = {}
    # for filename in os.listdir(out_dir):
    #     if filename.endswith('.pt'):
    #         all_feats.update(torch.load(os.path.join(out_dir, filename), map_location='cpu'))
    # torch.save(all_feats, os.path.join(data_dir, "scannet_mask3d_videofeats.pt"))

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)

