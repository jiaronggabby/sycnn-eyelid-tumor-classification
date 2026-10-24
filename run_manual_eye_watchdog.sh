#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_manual_eye_watchdog.sh --dataset-root PATH --output-root PATH --gpu IDS
  IDS is a comma-separated list of previously verified CUDA device IDs.
EOF
}

DATASET_ROOT=""
OUTPUT_ROOT=""
GPU_IDS=""
PYTHON_BIN="${PYTHON_BIN:-python}"
while (($#)); do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU_IDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

for value in DATASET_ROOT OUTPUT_ROOT GPU_IDS; do
  if [[ -z "${!value}" ]]; then
    echo "ERROR: missing --${value,,}" >&2
    usage
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="$SCRIPT_DIR/run_manual_eye_experiment.py"
command -v -- "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python executable not found: $PYTHON_BIN" >&2; exit 2; }
PYTHON_BIN="$(command -v -- "$PYTHON_BIN")"
SITE_PACKAGES="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for library_dir in "$SITE_PACKAGES/nvidia/cuda_nvrtc/lib" "$SITE_PACKAGES/nvidia/cuda_runtime/lib"; do
  if [[ -d "$library_dir" ]]; then
    LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
export LD_LIBRARY_PATH
[[ -f "$RUNNER" ]] || { echo "ERROR: missing runner: $RUNNER" >&2; exit 2; }
[[ -d "$DATASET_ROOT" ]] || { echo "ERROR: missing dataset root: $DATASET_ROOT" >&2; exit 2; }
DATASET_ROOT="$(cd -- "$DATASET_ROOT" && pwd -P)"
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
if [[ "$OUTPUT_ROOT" == "/" || "$DATASET_ROOT" == "$OUTPUT_ROOT" \
   || "$DATASET_ROOT" == "$OUTPUT_ROOT/"* || "$OUTPUT_ROOT" == "$DATASET_ROOT/"* ]]; then
  echo "ERROR: output root must be separate from and outside the dataset root" >&2
  exit 2
fi
mkdir -p -- "$OUTPUT_ROOT/logs"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
LOCK_PATH="$OUTPUT_ROOT/manual_eye_watchdog.lock"
COMPLETE_PATH="$OUTPUT_ROOT/manual_eye_experiment_complete.json"
STOP_PATH="$OUTPUT_ROOT/manual_eye_experiment.stop"
STATE_PATH="$OUTPUT_ROOT/logs/manual_eye_watchdog_state.log"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "ERROR: another watchdog already owns $LOCK_PATH" >&2
  exit 3
fi

attempt=0
runner_pid=""
runner_pgid=""
runner_process_wrapper=""
runner_pgid_file=""

stop_owned_group() {
  [[ -n "$runner_pid" && -n "$runner_pgid" ]] || return 0
  if [[ "$runner_pgid" != "$runner_pid" ]]; then
    echo "ERROR: refusing to signal unverifiable runner process group $runner_pgid" >&2
    return 1
  fi
  if ! ps -eo pgid=,sid= | awk -v id="$runner_pgid" '$1 == id && $2 == id { found = 1 } END { exit !found }'; then
    return 0
  fi
  if kill -0 -- "-$runner_pgid" 2>/dev/null; then
    kill -TERM -- "-$runner_pgid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 -- "-$runner_pgid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-$runner_pgid" 2>/dev/null || true
  fi
}

on_signal() {
  stop_owned_group || true
  if [[ -n "$runner_process_wrapper" ]]; then
    wait "$runner_process_wrapper" 2>/dev/null || true
  fi
  [[ -z "$runner_pgid_file" ]] || rm -f -- "$runner_pgid_file"
  exit 130
}
trap on_signal INT TERM

while :; do
  if [[ -f "$COMPLETE_PATH" ]]; then
    echo "Experiment complete: $COMPLETE_PATH"
    exit 0
  fi
  if [[ -f "$STOP_PATH" ]]; then
    echo "Stopped by explicit marker: $STOP_PATH" | tee -a "$STATE_PATH"
    exit 0
  fi

  attempt=$((attempt + 1))
  ATTEMPT_LOG="$OUTPUT_ROOT/logs/watchdog_attempt_${attempt}.log"
  printf '%s attempt=%s gpu=%s output=%s\n' "$(date -Is)" "$attempt" "$GPU_IDS" "$OUTPUT_ROOT" | tee -a "$STATE_PATH"
  runner_pgid_file="$OUTPUT_ROOT/logs/watchdog_attempt_${attempt}.pgid"
  setsid --wait bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' _ "$runner_pgid_file" \
    "$PYTHON_BIN" -B -u "$RUNNER" \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --gpu "$GPU_IDS" \
    --stage all >"$ATTEMPT_LOG" 2>&1 &
  runner_process_wrapper=$!
  for _ in {1..50}; do
    [[ -s "$runner_pgid_file" ]] && break
    kill -0 "$runner_process_wrapper" 2>/dev/null || break
    sleep 0.1
  done
  if [[ -s "$runner_pgid_file" ]]; then
    runner_pid="$(<"$runner_pgid_file")"
    runner_pgid="$(ps -o pgid= -p "$runner_pid" | tr -d '[:space:]')"
    runner_sid="$(ps -o sid= -p "$runner_pid" | tr -d '[:space:]')"
  else
    runner_pid=""
    runner_pgid=""
    runner_sid=""
  fi
  if [[ -z "$runner_pid" || "$runner_pgid" != "$runner_pid" || "$runner_sid" != "$runner_pid" ]]; then
    echo "ERROR: runner process group could not be verified; refusing automatic restart" | tee -a "$STATE_PATH" >&2
    wait "$runner_process_wrapper" || true
    runner_process_wrapper=""
    [[ -z "$runner_pgid_file" ]] || rm -f -- "$runner_pgid_file"
    runner_pgid_file=""
    exit 4
  fi

  completed_pid="$runner_pid"
  completed_pgid="$runner_pgid"
  set +e
  wait "$runner_process_wrapper"
  exit_code=$?
  set -e
  runner_process_wrapper=""

  if [[ -f "$COMPLETE_PATH" ]]; then
    runner_pid=""
    runner_pgid=""
    rm -f -- "$runner_pgid_file"
    runner_pgid_file=""
    printf '%s status=complete attempt=%s\n' "$(date -Is)" "$attempt" | tee -a "$STATE_PATH"
    exit 0
  fi

  # The process group is scoped to the runner and its own seed children.
  if [[ "$exit_code" -eq 0 ]]; then
    runner_pid=""
    runner_pgid=""
    rm -f -- "$runner_pgid_file"
    runner_pgid_file=""
    echo "Runner exited successfully without a completion marker; refusing to claim completion" | tee -a "$STATE_PATH" >&2
    exit 5
  fi
  runner_pid="$completed_pid"
  runner_pgid="$completed_pgid"
  stop_owned_group || exit 6
  runner_pid=""
  runner_pgid=""
  rm -f -- "$runner_pgid_file"
  runner_pgid_file=""
  printf '%s status=restart exit_code=%s attempt=%s checkpoint_resume_expected=true\n' \
    "$(date -Is)" "$exit_code" "$attempt" | tee -a "$STATE_PATH"
  sleep "${MANUAL_EYE_RESTART_DELAY_SECONDS:-15}"
done
