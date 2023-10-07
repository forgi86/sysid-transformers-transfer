# encoder-decoder simulation transformer on LTI systems
run train_sim_lin --log-wandb

# encoder-decoder simulation transformer pre-trained on LTI systems adapted on WH systems
run train_sim_wh  --init-from pretrained --in-file ckpt_sim_lin --fixed-lr --lr 1e-4 --warmup-iters 0 --max-iters 5_000_000 --log-wandb