#!/usr/bin/env bash
set -euo pipefail

cd /home/fey006/Project1

source /opt/conda/etc/profile.d/conda.sh
conda activate rapidfire

export OPENAI_API_KEY="$(cat /home/fey006/api-key.txt)"
export OPENAI_BASE_URL="https://tritonai-api.ucsd.edu/v1"
export JUDGE_BASE_URL="https://tritonai-api.ucsd.edu/v1"
export JUDGE_MODEL="claude-sonnet-4-6"
export RAG_MODEL="api-gpt-oss-120b"
export RF_MLFLOW_PORT="${RF_MLFLOW_PORT:-8854}"

run_one() {
  local config_id="$1"
  local dir_name="$2"
  local out_dir="rapidfire_experiments_validation/${dir_name}"
  local answer_file="${out_dir}/validation_answers.json"

  if [[ -f "${answer_file}" ]]; then
    echo "[SKIP] ${config_id} already has ${answer_file}"
    return 0
  fi

  if [[ -d "${out_dir}" ]]; then
    echo "[CLEAN] Removing partial output for ${config_id}: ${out_dir}"
    rm -rf "${out_dir}"
  fi

  echo "[RUN] ${config_id}: ${dir_name}"
  python3 rapidfire_experiments_validation.py \
    --mode tfidf-single \
    --tfidf-config-id "${config_id}" \
    --experiment-name project1-validation-tfidf
}

run_one T06 T06_tfidf_512_96_k1_title_no_related_expand0
run_one T07 T07_tfidf_320_64_k1_title_related_expand0
run_one T08 T08_tfidf_420_80_k1_title_related_expand0
run_one T09 T09_tfidf_512_96_k1_title_related_expand0
run_one T10 T10_tfidf_420_80_k1_title_related_expand5
run_one T11 T11_tfidf_420_80_k1_title_related_expand10
run_one T12 T12_tfidf_420_80_k3_title_related_expand5
run_one T13 T13_tfidf_420_80_k5_title_related_expand5

echo "[DONE] Remaining validation TF-IDF configs finished."
