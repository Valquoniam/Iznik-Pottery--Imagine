#!/bin/bash

sh_file_name="script_inference.sh"
gpu="0"
config="tiles.yml"
guid="flower"
test_step=100    # if large, it takes long time.
dt_lambda=1.0   # hyperparameter for dt_lambda. This is the method that will appear in the next paper.

CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
                        --config $config                                  \
                        --exp ./runs/$guid                                \
                        --edit_attr $guid                                  \
                        --do_train 1                                        \
                        --do_test 1                                         \
                        --n_train_img 1000                                   \
                        --n_test_img 32                                     \
                        --n_iter 5                                          \
                        --bs_train 1                                        \
                        --t_0 999                                           \
                        --n_inv_step 50                                     \
                        --n_train_step 50                                   \
                        --n_test_step $test_step                            \
                        --get_h_num 1                                       \
                        --train_delta_block                                 \
                        --sh_file_name $sh_file_name                       \
                        --save_x0                                           \
                        --use_x0_tensor                                     \
                        --hs_coeff_delta_h 1.0                              \
                        --dt_lambda $dt_lambda                              \
                        --custom_train_dataset_dir "test_images/afhq/train"                \
                        --custom_test_dataset_dir "test_images/afhq/test"                  \
                        --add_noise_from_xt                                 \
                        --lpips_addnoise_th 1.2                             \
                        --lpips_edit_th 0.33                                \
                        --sh_file_name "script_inference.sh"                \
                        --model_path "pretrained/afhqdog_p2.pt"
                         
                        # if you did not compute lpips, use it.
                        # --user_defined_t_edit 500                           \
                        # --user_defined_t_addnoise 200                       \

