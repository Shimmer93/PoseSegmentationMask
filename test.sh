GPUS=8 CPUS_PER_TASK=2 ./tools/slurm_test.sh gpu-share mmpose_test configs/body_2d_keypoint/topdown_psm/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py logs/best_coco_AP_epoch_121.pth