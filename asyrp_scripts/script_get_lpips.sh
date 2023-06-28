#!/bin/bash

sh_file_name="script_get_lpips.sh"
gpu="0"
config="tiles.yml"     # for custom.yml, you need to set custom_train_dataset_dir and custom_test_dataset_dir. If not, vice versa.
guid="floral"          # we don't use it but need to run asyrp.py
inv_step=50           # if large, it takes long time.

CUDA_VISIBLE_DEVICES=$gpu python asyrp.py --lpips                            \
                        --config $config                                    \
                        --exp ./runs/tmp                                    \
                        --edit_attr $guid                                   \
                        --do_train 1                                        \
                        --do_test 1                                         \
                        --n_train_img 999                                   \
                        --n_test_img 32                                     \
                        --t_0 999                                           \
                        --n_inv_step $inv_step                              \
                        --sh_file_name "script_get_lpips.sh"                \
                        --model_path "../iznik/Iznik-Pottery--Imagine/logs/ddpm_unet/tiles_orig_unet_size_64_steps_1000_lr_0.001_aug_flip,rotate,symmetry/lightning_logs/version_0/checkpoints/epoch=956.ckpt" \

#                        --custom_train_dataset_dir "test_images/celeba/train"       \
#                        --custom_test_dataset_dir "test_images/celeba/test"         \
