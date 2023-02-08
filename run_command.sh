# ===== training CIFAR-10 =======
# 512
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 512 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmpp=1 --batch 512 --aug_dim 512
# 8192
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 8192 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmpp=1 --batch 512 --aug_dim 8192
# 3072000
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name 3072000 \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --pfgmpp=1 --batch 512 --aug_dim 3072000
# edm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name edm_ncsnpp \
  --data=datasets/cifar10-32x32.zip --cond=0 --arch=ncsnpp \
  --batch 512

# ===== generate =======
# 512
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/512 \
  --ckpt 100000 --pfgmpp=1 --aug_dim=512
# 8192
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/8192 \
  --ckpt 100000 --pfgmpp=1 --aug_dim=8192
# 3072000
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/3072000 \
  --ckpt 100000 --pfgmpp=1 --aug_dim=3072000
# edm
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/edm_ncsnpp \
  --ckpt 100000


# ===== fid evaluation =====
# 512
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/512 --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_512_8GPU.txt
# 8192
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/8192 --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_8192_8GPU.txt
# 3072000
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/3072000 --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_3072000_8GPU.txt
# edm
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/edm_ncsnpp --ref=fid-refs/cifar10-32x32.npz --num 50000 --ckpt 100000 --gen_seed 1 > res_edm.txt


# ================ Train FFHQ ================
# 128
torchrun --standalone --nproc_per_node=8 train.py \
  --outdir=training-runs --name ffhq_128 \
  --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp \
  --pfgmpp=1 --batch 256 --aug_dim 128 \
  --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
# generate
torchrun --standalone --nproc_per_node=8 generate.py \
  --seeds=0-49999 --outdir=./training-runs/ffhq_128 \
  --ckpt 150000 --pfgmpp=1 --aug_dim=128 --steps=40
# fid evaluation
torchrun --standalone --nproc_per_node=8 fid.py calc --images=training-runs/ffhq_128 --ref=fid-refs/ffhq-64x64.npz --num 50000 --ckpt 150000 --gen_seed 1 > res_ffhq_ddpmpp_128.txt

# train edm model in the edm_pfgm code base
torchrun --standalone --nproc_per_node=8 train.py   --outdir=training-runs --name ffhq_edm_ddpmpp   --data=datasets/ffhq-64x64.zip --cond=0 --arch=ddpmpp   --batch 256   --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
