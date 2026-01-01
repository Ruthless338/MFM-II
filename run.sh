# 将 ISIC 全量训练数据 10017 张(600*450) resize (256*256) -> VAE encoder 得到 latents [4,32,32]
HF_ENDPOINT=https://hf-mirror.com python scripts/precompute_latents.py

# 使用全量数据 8:2 得到的 train_split.csv 和 val_split.csv 训练 FM
HF_ENDPOINT=https://hf-mirror.com python scripts/train_fm.py

# 推理 FM 模型，检查生成效果
HF_ENDPOINT=https://hf-mirror.com python scripts/inference_fm.py

# 类内随机组合(train_split.csv) -> FlowODESolver Inversion(Real -> Noise) -> Circle Interpolation -> FM -> VAE decoder -> Synthetic Data, 所有类别都扩增5倍
HF_ENDPOINT=https://hf-mirror.com python scripts/generate_aug.py

# train classifier
# 实验名称	训练数据组成
# Exp 1: Baseline	100% Real Data
# Exp 2: Ours (Full)	100% Real + Flow Matching Circle Interpolation Synthetic Data
# Exp 3: Baseline 10% Real Data
# Exp 4: Few-shot	仅用 10% Real Data + FM Circle Interpolation Synthetic Data	极度缺数据
# Exp 5: Few-shot	仅用 10% Real Data + FM Linear Interpolation Synthetic Data	线性插值，对比我的方法优越性


python scripts/train_classifier.py \
  --exp_name "Exp1_Baseline_Real100" \
  --real_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input" \
  --train_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/train_split.csv" \
  --val_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv" \
  --devices 4 \
  --batch_size 512 \
  --epochs 50

python scripts/train_classifier.py \
  --exp_name "Exp2_Ours_FM_Augmented" \
  --use_synthetic \
  --syn_data_dir "/mnt/data0/cyb/ISIC/isic_augmented_256" \
  --real_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input" \
  --train_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/train_split.csv" \
  --val_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv" \
  --devices 4 \
  --batch_size 512 \
  --epochs 50

# 为 Exp3 准备 10% Real Image 合成的Augmentation数据，不能使用全量数据生成的合成数据，这样会泄露数据导致实验不严谨
# 先将 train_split.scv 做 10% 切分
# ------------------------------
# Original Train size: 8012
# Few-shot (10%) size: 801
# Unused size:         7211
# ------------------------------
# Class distribution in Few-shot set:
#   MEL: 89
#   NV: 536
#   BCC: 41
#   AKIEC: 26
#   BKL: 88
#   DF: 9
#   VASC: 12
# ------------------------------
# fewshot data 中 DF 等类别数量过少，类内随机组合最少只有 36 种，故做 Seed Amplification( Flip/Rotate-> num * 4，即 89 + 536 + 41*5 + 26*5 + 88 + 9*5 + 12*5 = 1153 )
# 再VAE encoder -> latents
HF_ENDPOINT=https://hf-mirror.com python scripts/precompute_latents.py

# fewshot aug
HF_ENDPOINT=https://hf-mirror.com python scripts/generate_aug_fewshot.py \
  --ckpt /home/chenyibiao/MFM-II/checkpoints/isic_fm_dit_v2/epoch=384-val/loss=1.7222.ckpt \
  --data_dir /mnt/data0/cyb/ISIC/isic2018_fewshot_10_latents_256 \
  --train_csv /mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv \
  --output_dir /mnt/data0/cyb/ISIC/isic_fewshot_10_augmented_256 \
  --target_count 536 \
  --device 0 \
  --interpolation_mode Slerp

python scripts/train_classifier.py \
  --exp_name "Exp3_Baseline_Real10_amplified" \
  --real_img_dir "/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw" \
  --train_csv "/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv" \
  --val_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv" \
  --val_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input" \
  --devices 4 \
  --batch_size 512 \
  --epochs 50

python scripts/train_classifier.py \
  --exp_name "Exp4_FewShot_10_with_Aug" \
  --use_synthetic \
  --syn_data_dir "/mnt/data0/cyb/ISIC/isic_fewshot_10_augmented_256" \
  --real_img_dir "/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw" \
  --train_csv "/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv" \
  --val_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv" \
  --val_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input" \
  --devices 4 \
  --batch_size 512 \
  --epochs 50

# inference_classifer
python scripts/inference_classifier.py \
  --ckpt_path "/home/chenyibiao/MFM-II/checkpoints/classifier/Exp1_Baseline_Real100/best-epoch=49-val/f1_macro=0.7536.ckpt" \
  --test_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_Input" \
  --test_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv" \
  --output_dir "results/inference_classifier/ISIC/Exp1"

python scripts/inference_classifier.py \
  --ckpt_path "/home/chenyibiao/MFM-II/checkpoints/classifier/Exp2_Ours_FM_Augmented/best-epoch=46-val/f1_macro=0.7917.ckpt" \
  --test_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_Input" \
  --test_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv" \
  --output_dir "results/inference_classifier/ISIC/Exp2"

python scripts/inference_classifier.py \
  --ckpt_path "/home/chenyibiao/MFM-II/checkpoints/classifier/Exp3_Baseline_Real10_amplified/best-epoch=40-val/f1_macro=0.2471.ckpt" \
  --test_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_Input" \
  --test_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv" \
  --output_dir "results/inference_classifier/ISIC/Exp3"

python scripts/inference_classifier.py \
  --ckpt_path "/home/chenyibiao/MFM-II/checkpoints/classifier/Exp4_FewShot_10_with_Aug/best-epoch=48-val/f1_macro=0.4527.ckpt" \
  --test_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_Input" \
  --test_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv" \
  --output_dir "results/inference_classifier/ISIC/Exp4"


# Exp5 Linear interpolation
HF_ENDPOINT=https://hf-mirror.com python scripts/generate_aug_fewshot.py \
  --ckpt /home/chenyibiao/MFM-II/checkpoints/isic_fm_dit_v2/epoch=384-val/loss=1.7222.ckpt \
  --data_dir /mnt/data0/cyb/ISIC/isic2018_fewshot_10_latents_256 \
  --train_csv /mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv \
  --output_dir /mnt/data0/cyb/ISIC/isic_fewshot_10_augmented_256_Linear \
  --target_count 536 \
  --interpolation_mode Linear \
  --device 0
  
python scripts/train_classifier.py \
  --exp_name "Exp5_FewShot_10_with_Aug_Linear" \
  --use_synthetic \
  --syn_data_dir "/mnt/data0/cyb/ISIC/isic_fewshot_10_augmented_256_Linear" \
  --real_img_dir "/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw" \
  --train_csv "/mnt/data0/cyb/ISIC/isic2018_fewshot_10_raw/train_amplified.csv" \
  --val_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_GroundTruth/val_split.csv" \
  --val_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Training_Input" \
  --devices 3 \
  --batch_size 512 \
  --epochs 50

python scripts/inference_classifier.py \
  --ckpt_path "/home/chenyibiao/MFM-II/checkpoints/classifier/Exp5_FewShot_10_with_Aug_Linear/best-epoch=29-val/f1_macro=0.4994.ckpt" \
  --test_img_dir "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_Input" \
  --test_csv "/mnt/data0/cyb/ISIC/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv" \
  --output_dir "results/inference_classifier/ISIC/Exp5"