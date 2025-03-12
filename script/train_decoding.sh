nohup python3 train_diffusion.py --model_name BrainTranslator \
    --task_name task1_task2_taskNRv2 \
    --two_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 50 \
    --num_epoch_step2 20 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    --cuda cuda:1 \
    -s ./checkpoints >ldm_train.log &

nohup python3 pretrain_autoencoder.py \
    --task_name task1_task2_taskNRv2 \
    --num_epoch 50 \
    --train_input EEG \
    -lr 0.00002 \
    -b 32 \
    --cuda cuda:1 \
    -s ./checkpoints/autoencoder >