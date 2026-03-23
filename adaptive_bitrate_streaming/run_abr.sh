python run_abr.py  \
                  --test \
                  --frozen \
                  --state-use-self-attention \
                  --grad-accum-steps 32 \
                  --seed 666 \
                  --plm-type llama \
                  --plm-size base \
                  --rank 128 \
                  --device cuda:0 \
                  --state-feature-dim 512 \
                  --w 20 \
                  --gamma 1. \
                  --lr 5e-5 \
                  --warmup-steps 2000 \
                  --num-epochs 70 \
                  --eval-per-epoch 2 \
                  --target-return-scale 1 \
                  --save-checkpoint-per-epoch 40 \
                  --state-attn-hidden-dim 2048 \
                  #--video video2 \
                  #--trace hsr-test \
                  #--fusion-method mamba 