#!/bin/bash
# 步骤3: 批量训练消融实验
# 用法: bash ablation/3_run_experiments.sh [--dry-run]
#   --dry-run  仅打印命令, 不执行训练

set -e

# ============================================================
# 配置 - 按需修改
# ============================================================
NUM_GPUS=1
SEEDS=(42 123 999)

# 实验配置: "配置名:config_path"
EXPERIMENTS=(
  "4fusions:ablation/configs/san_vit_b16_voc_4fusions.yaml"
  "6fusions:ablation/configs/san_vit_b16_voc_6fusions.yaml"
)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# ============================================================
# 环境检查
# ============================================================
echo "============================================"
echo "  SAN 消融实验 - 批量训练"
echo "============================================"
echo ""

# 确保 wandb 关闭
export WANDB_MODE=offline

# 检查基础环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || {
  echo "ERROR: PyTorch 环境不可用"
  exit 1
}

# 检查数据准备
SPLIT_FILE="ablation/output/splits/train.txt"
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "数据集划分未完成, 先运行: python ablation/1_prepare_splits.py"
  exit 1
fi
echo "✓ 数据集划分已就绪"

# 注册数据集 (必须在训练前执行)
python -c "import ablation" 2>/dev/null && echo "✓ 数据集注册成功" || {
  echo "ERROR: 数据集注册失败, 检查 ablation/__init__.py"
  exit 1
}
echo ""

# ============================================================
# 训练循环
# ============================================================
TOTAL=$(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))
CURRENT=0
START_TIME=$(date +%s)

for exp_entry in "${EXPERIMENTS[@]}"; do
  EXP_NAME="${exp_entry%%:*}"
  CONFIG="${exp_entry##*:}"

  for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    OUTPUT_DIR="ablation/output/${EXP_NAME}_seed${SEED}"
    TIMELOG="${OUTPUT_DIR}/train_time.txt"

    echo "[${CURRENT}/${TOTAL}] ${EXP_NAME} | seed=${SEED}"
    echo "  Config: ${CONFIG}"
    echo "  Output: ${OUTPUT_DIR}"

    if $DRY_RUN; then
      echo "  [DRY RUN] python train_net.py --config-file ${CONFIG} --num-gpus ${NUM_GPUS} OUTPUT_DIR ${OUTPUT_DIR} SEED ${SEED}"
    else
      mkdir -p "${OUTPUT_DIR}"

      T0=$(date +%s)
      python train_net.py \
        --config-file "${CONFIG}" \
        --num-gpus ${NUM_GPUS} \
        OUTPUT_DIR "${OUTPUT_DIR}" \
        SEED ${SEED} \
        2>&1 | tee "${OUTPUT_DIR}/train_log.txt"

      T1=$(date +%s)
      ELAPSED=$((T1 - T0))
      echo "train_time_seconds=${ELAPSED}" > "${TIMELOG}"
      echo "train_time_hours=$(echo "scale=2; ${ELAPSED}/3600" | bc)" >> "${TIMELOG}"
      echo "  完成! 耗时: ${ELAPSED}s ($(echo "scale=1; ${ELAPSED}/3600" | bc)h)"
    fi
    echo ""
  done
done

# ============================================================
# 汇总
# ============================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
echo "============================================"
echo "  全部训练完成!"
echo "  总耗时: ${TOTAL_ELAPSED}s ($(echo "scale=1; ${TOTAL_ELAPSED}/3600" | bc)h)"
echo "============================================"
