#!/bin/bash

#PPO_WEIGHTS=""
PPO_WEIGHTS="${PPO_WEIGHTS:?Set PPO_WEIGHTS}"
SPLIT="pufferhard"
OUTPUT_DIR="${PUFFER_EXP_ROOT:-experiments}"

submit () {
  local ego="$1"
  local other="$2"
  local map_ids="$3"

  args=(--split "$SPLIT" --output-dir $OUTPUT_DIR --map-ids $map_ids)

  # ego planner
  if [[ "$ego" == "ppo" ]]; then
    args+=(--ego-planner ppo --ego-weights "$PPO_WEIGHTS")
  elif [[ "$ego" == "cv" ]]; then
    args+=(--ego-planner constant_velocity)
  elif [[ "$ego" == "pdm" ]]; then
    args+=(--ego-planner pdm)
  elif [[ "$ego" == "cem" ]]; then
    args+=(--ego-planner cem)
  elif [[ "$ego" == "idm" ]]; then
    args+=(--ego-planner idm)
  fi

  # other planner
  if [[ "$other" == "ppo" ]]; then
    args+=(--other-planner ppo --other-weights "$PPO_WEIGHTS")
  elif [[ "$other" == "cv" ]]; then
    args+=(--other-planner pdm)
  elif [[ "$other" == "pdm" ]]; then
    args+=(--other-planner cem)
  elif [[ "$other" == "cem" ]]; then
    args+=(--other-planner cem)
  elif [[ "$other" == "idm" ]]; then
    args+=(--other-planner idm)
  fi

  JOB_NAME="viz_${ego}_vs_${other}"

  echo "Submitting: ${JOB_NAME}"
  sbatch --job-name="$JOB_NAME" run_single_evaluation.sh "${args[@]}"
}

# ------------------------------------------------------------
# All combinations
# ------------------------------------------------------------

submit ppo idm 3,7,14,19,20,25,28,29,30,47,65,69,75,81,91,95,103,121,127,133,143,161,162,163,166,168,170,171,172,177,180,182,187,195,198,209,232,233,236,240,242,244,248,253,273,281,282,287,298,304,307,310,311,319,323,326,327,330,331,333,340,346,350,351,352,358,365,367,369,374,375,378,391,395,402,409,412,430,440,452,453,456,471,472,481,491,492,501,502,504,508,516,521,528,539,542,546,552,558,563,567,568,575,580,590,591,592,597,603,607,613,616,621,633,641,643,648,654,657,665,680,686,690,695,703,711,712,720,721,723,729,730,731,738,740,745,748,768,772,775,776,783,791,792,794,797,799,801,810,817,820,823,831,839,841,848,850,851,864,869,886,887,898,900,905,914,927,928,929,932,940,944,945,947,949,958,961,962,964,973,992,995,997