#  python scripts/feature_fusion/scannet_videofeat_odin.py \
#     --data_dir /mnt/ssd/liuchao/odin/scannet/frames_square_highres \
#     --output_dir /mnt/ssd/liuchao/Chat-Scene/odin_videofeats \
#     --data_mode odin

#  python scripts/feature_fusion/scannet_videofeat.py \
#     --data_dir /mnt/ssd/liuchao/Chat-Scene/data \
#     --output_dir /mnt/ssd/liuchao/Chat-Scene/mask3d_videofeats2 \
#     --data_mode mask3d

CUDA_VISIBLE_DEVICES=1 python scripts/feature_fusion/scannet_videofeat_sam.py \
    --data_dir /mnt/ssd/liuchao/Chat-Scene/data \
    --output_dir ./show_sam_project \
    --data_mode mask3d

