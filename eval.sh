#!/bin/bash
# 步骤4: 在Test集评估所有checkpoint + 测量推理速度
# 用法: bash ablation/4_eval_all.sh [--dry-run]

set -e

NUM_GPUS=1
SEEDS=(42 123 999)
TEST_DATASET="voc_ablation_test"

# 实验定义: "名称:config_path"
EXPERIMENTS=(
  "4fusions:ablation/configs/san_vit_b16_voc_4fusions.yaml"
  "6fusions:ablation/configs/san_vit_b16_voc_6fusions.yaml"
)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

export WANDB_MODE=offline

# 注册数据集
python -c "import ablation" 2>/dev/null || {
  echo "ERROR: 数据集注册失败"
  exit 1
}

# 查找checkpoint
find_checkpoint() {
  local train_dir="$1"
  for ckpt in "$train_dir"/model_final.pth "$train_dir"/model_*.pth; do
    if [[ -f "$ckpt" ]]; then
      echo "$ckpt"
      return 0
    fi
  done
  return 1
}

TOTAL=$(( ${#EXPERIMENTS[@]} * ${#SEEDS[@]} ))
CURRENT=0

for exp_entry in "${EXPERIMENTS[@]}"; do
  EXP_NAME="${exp_entry%%:*}"
  CONFIG="${exp_entry##*:}"

  for SEED in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    TRAIN_DIR="ablation/output/${EXP_NAME}_seed${SEED}"
    EVAL_DIR="${TRAIN_DIR}/test_eval"

    echo "[${CURRENT}/${TOTAL}] 评估 ${EXP_NAME} seed=${SEED}"

    # 查找checkpoint
    CKPT=$(find_checkpoint "$TRAIN_DIR")
    if [[ $? -ne 0 ]]; then
      echo "  SKIP: 未找到模型权重 (${TRAIN_DIR}/model_*.pth)"
      continue
    fi
    echo "  Checkpoint: ${CKPT}"

    if $DRY_RUN; then
      echo "  [DRY RUN] python train_net.py --eval-only --config-file ${CONFIG} ..."
    else
      mkdir -p "${EVAL_DIR}"

      # 测试集评估 + 计时
      T0=$(date +%s%N)
      python train_net.py \
        --eval-only \
        --config-file "${CONFIG}" \
        --num-gpus ${NUM_GPUS} \
        MODEL.WEIGHTS "${CKPT}" \
        DATASETS.TEST "('${TEST_DATASET}',)" \
        OUTPUT_DIR "${EVAL_DIR}" \
        2>&1 | tee "${EVAL_DIR}/eval_log.txt"
      T1=$(date +%s%N)

      ELAPSED_MS=$(( (T1 - T0) / 1000000 ))
      ELAPSED_S=$(echo "scale=1; ${ELAPSED_MS}/1000" | bc)
      TEST_IMAGES=$(python -c "
from detectron2.data import DatasetCatalog
print(len(DatasetCatalog.get('${TEST_DATASET}')))
" 2>/dev/null || echo 1449)
      FPS=$(echo "scale=2; ${TEST_IMAGES} / ${ELAPSED_S}" | bc 2>/dev/null || echo "N/A")

      echo "eval_time_ms=${ELAPSED_MS}" > "${EVAL_DIR}/eval_time.txt"
      echo "eval_fps=${FPS}" >> "${EVAL_DIR}/eval_time.txt"
      echo "eval_num_images=${TEST_IMAGES}" >> "${EVAL_DIR}/eval_time.txt"
      echo "  完成! 耗时: ${ELAPSED_S}s, FPS: ${FPS}"
    fi
    echo ""
  done
done

echo "============================================"
echo "  全部评估完成!"
echo "============================================"
