conda activate sysid-t-t

for i in $(seq 1 50)
do
    python train_sim_paper.py --max-iters=10000 --seed=$i --init-from=pretrained --model-dir=models --in-file=ckpt_sim_wh --out-file=model_pwh_seed_$i --threads=1 --cuda-device=cuda:2 --eval-interval=100 --nx=50 --fixed-system --log-wandb
    wait
done