#!/usr/bin/env bash
#

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
echo "$SCRIPT_PATH"

DATASETNAMES=("synthesis_cancerpathway_GEX"
          )

MODELS=("transformer")

ENDPOINT_INDEX=("SLNM@status")

CUDA_IDX=0

for dataset in "${DATASETNAMES[@]}"; do
  for model   in "${MODELS[@]}"; do
    for ep in "${ENDPOINT_INDEX[@]}"; do
      for seed in {0..0}; do
        # run each fold
        for k in {0..0}; do
          echo " Training dataset:$dataset model:$model ep:$ep repeat:$seed fold:$k"
          logPath="$SCRIPT_PATH""/log"
          logName="/""$dataset""_""$ep""_""$model""_""$seed""_fold""$k"".txt"
          mkdir -p "$(dirname "$logPath$logName")"
          echo $logName
          output_dir="output/""${dataset}""/""${ep}""/""${model}""/""${seed}""/""${k}"
          rm -rf ${output_dir}
          if [ "$model" == "xgboost" ]; then
            nohup python ClinGEX-DL/bin/${model}_.py ClinGEX-DL/configs/gex_${model}_.toml \
            --dataset ${dataset} \
            --output ${output_dir} --endpoint ${ep} --KFold ${k} \
            --repeat ${seed} >$logPath$logName 2>&1 &
          else
            CUDA_VISIBLE_DEVICES=${CUDA_IDX} nohup python ClinGEX-DL/bin/${model}.py ClinGEX-DL/configs/gex_${model}.toml \
            --dataset ${dataset} \
            --output ${output_dir} --endpoint ${ep} --KFold ${k} \
            --repeat ${seed} >$logPath$logName 2>&1 &
          fi
          wait
        done
      done
    done
  done
done

