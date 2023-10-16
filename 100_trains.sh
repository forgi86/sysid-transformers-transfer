conda activate sysid-t-t

for i in $(seq 1 50)
do
    python train_sim_wh.py --max-iters=10000 --seed=$i --init-from=pretrained --model-dir=models --in-file=ckpt_sim_wh --threads=1 --cuda-device=cuda:3 --eval-interval=150 --fixed-system --log-wandb
    wait
done