CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --device cuda \
    --dataset NUSWIDE \
    --notes "nus_wide/16bits" \
    --trainable_layer_num 0 \
    --M 2 \
    --feat_dim 32 \
    --T 0.5 \
    --tau_cqc 0.5 \
    --hp_beta 5e-3 \
    --softmax_temp 15.0 \
    --clip_r 1.1 \
    --init_neg_curvs 1.0 \
    --full_hyperpq \
    --disable_writer \
    --epoch_num 50 \
    --clus_mode "hier_clus" \
    --num_clus_list "100,50,25" \
    --warmup_epoch 3 \
    --eval_interval 1 \
    --topK 5000 \
    --checkpoint_dir ./checkpoints/nus_wide/16bits \

    
