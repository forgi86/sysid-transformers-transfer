# encoder-decoder simulation transformer on LTI systems
run train_sim_lin --log-wandb

# encoder-decoder simulation transformer pre-trained on LTI systems adapted on WH systems
run train_sim_wh  --init-from pretrained --in-file ckpt_sim_lin --fixed-lr --lr 1e-4 --warmup-iters 0 --max-iters 5_000_000 --log-wandb

# encoder-decoder simulation transformer transfer from 100 to 500, starting from pre-trained
run train_sim_wh --init-from pretrained --in-file ckpt_sim_wh --seq-len-new 200 --fixed-lr --lr 1e-4 --log-wandb

# encoder-decoder simulation transformer transfer 200, starting from scratch
run train_sim_wh --init-from scratch --seq-len-new 200 --log-wandb

# encoder-decoder simulation transformer transfer 10, starting from scratch
run train_sim_wh --init-from scratch --seq-len-new 10 --out-file ckpt_sim_wh_10_scratch --log-wandb

# encoder-decoder simulation transformer transfer 100, starting from scratch
run train_sim_wh --init-from scratch --seq-len-new 100 --out-file ckpt_sim_wh_100_scratch --log-wandb # interrupted

# encoder-decoder simulation transformer transfer from 100 to 1000, starting from pre-trained
run train_sim_wh --init-from pretrained --in-file ckpt_sim_wh --seq-len-new 1000 --fixed-lr --lr 1e-4 --out-file ckpt_sim_wh_1000_pre --log-wandb

# encoder-decoder simulation transformer transfer from scratch on pwh
run train_sim_wh --init-from scratch --out-file ckpt_sim_pwh_100 --log-wandb