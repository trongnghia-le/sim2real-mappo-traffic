#!/usr/bin/env bash
# =============================================================================
# GCP Spot VM startup script for Sim2Real-MAPPO-Traffic (Ubuntu 22.04)
#
# Idempotent and safe to re-run on Spot preemption reboots:
#   - Ensures /snap/bin is on PATH and installs google-cloud-cli from Google's
#     apt repo if gsutil is still missing.
#   - Installs minimal pinned Python training+eval dependencies (CPU-only
#     PyTorch wheel from the PyTorch CPU index + scientific/RL stack + pinned
#     eclipse-sumo==1.24.0, traci==1.24.0, sumolib==1.24.0 — no ultralytics,
#     opencv, streamlit, torchvision, or CUDA).
#   - Clones or updates the public repo at the metadata-specified branch and
#     idempotently generates any missing canonical demand files using the
#     venv's duarouter if a fresh clone lacks them.
#   - Runs scripts/parallel_launcher.py under nohup with a fixed --campaign-id
#     and per-shard manifest/log naming so restarts resume completed jobs
#     (bench.json) and in-progress runs (in-place last_model.pt resume).
#   - Every 30 minutes runs `gsutil -m rsync -r` of results/ and models/ to
#     gs://$BUCKET/<campaign-id>/.
#   - On any startup failure, uploads startup.log best-effort and shuts the VM
#     down when auto-shutdown=true.
#   - On launcher completion (or non-zero exit with zero completed jobs),
#     performs a final sync and runs `shutdown -h now` when auto-shutdown=true.
#
# Instance metadata attributes (all optional except `bucket`):
#   bucket           (required) GCS bucket name (without gs:// prefix)
#   campaign-id      fixed campaign identifier         (default: revision130)
#   branch           git branch to checkout            (default: revision/schema-1.3.0)
#   max-workers      concurrent training jobs          (default: nproc / 2)
#   preset           launcher --preset value           (default: revision130)
#   tier             launcher --tier value (1|2|all)   (default: all)
#   shard            optional launcher --shard i/n     (default: "")
#   stagger          seconds between worker spawns     (default: 15)
#   total-timesteps  optional --total-timesteps override (default: "" -> preset default 300000)
#   extra-args       train_ppo.py passthrough after -- (default: --no-libsumo)
#   sync-interval-s  seconds between GCS rsyncs        (default: 1800)
#   auto-shutdown    shutdown VM on completion/failure (default: true)
# =============================================================================
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:${PATH:-}"

REPO_URL="https://github.com/tronghia256-EdgeAI/Sim2Real-MAPPO-Traffic.git"
SUMO_VERSION="1.24.0"
WORK_ROOT="/opt/sim2real-mappo"
REPO_DIR="${WORK_ROOT}/repo"
VENV_DIR="${WORK_ROOT}/.venv"
DEPS_MARKER="${VENV_DIR}/.deps_installed_min_sumo_${SUMO_VERSION}"
LOG_DIR="/var/log/sim2real-mappo"
STARTUP_LOG="${LOG_DIR}/startup.log"
SUPERVISOR_LOG="${LOG_DIR}/supervisor.log"
SUPERVISOR_PIDFILE="/var/run/sim2real-supervisor.pid"
SUPERVISOR_SCRIPT="/usr/local/bin/sim2real-supervisor.sh"
ENV_FILE="/etc/sim2real-campaign.env"

# Defaults initialized early so the ERR/EXIT trap can safely inspect them even
# if a failure happens before metadata parsing finishes.
BUCKET=""
CAMPAIGN_ID="revision130"
SHARD=""
SHARD_TAG=""
MANIFEST_NAME="campaign_manifest.json"
GCS_STARTUP_LOG="startup.log"
GCS_SUPERVISOR_LOG="supervisor.log"
GCS_LAUNCHER_LOG="launcher.log"
AUTO_SHUTDOWN="true"
STARTUP_OK=0

mkdir -p "${WORK_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${STARTUP_LOG}") 2>&1

log() {
  printf '[%s] [startup] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

on_startup_err() {
  local rc=$?
  local line="${1:-unknown}"
  log "ERROR: startup.sh failed at line ${line} (rc=${rc})."
}

on_startup_exit() {
  local rc=$?
  trap - EXIT ERR
  set +e
  if [ "${STARTUP_OK}" -eq 1 ] && [ "${rc}" -eq 0 ]; then
    return 0
  fi
  log "Startup abort handler triggered (rc=${rc}, startup_ok=${STARTUP_OK}); uploading log best-effort..."
  if [ -n "${BUCKET}" ] && [ -n "${CAMPAIGN_ID}" ] && command -v gsutil >/dev/null 2>&1; then
    if [ -f "${STARTUP_LOG}" ]; then
      gsutil cp "${STARTUP_LOG}" "gs://${BUCKET}/${CAMPAIGN_ID}/${GCS_STARTUP_LOG}" || true
    fi
    if [ -f "${SUPERVISOR_LOG}" ]; then
      gsutil cp "${SUPERVISOR_LOG}" "gs://${BUCKET}/${CAMPAIGN_ID}/${GCS_SUPERVISOR_LOG}" || true
    fi
  fi
  if [ "${AUTO_SHUTDOWN}" = "true" ]; then
    log "auto-shutdown=true: shutting down VM after startup failure."
    shutdown -h now || true
  fi
  exit "${rc}"
}

trap 'on_startup_err ${LINENO}' ERR
trap 'on_startup_exit' EXIT

get_meta() {
  local key="$1"
  local default_val="${2:-}"
  local val=""
  if val="$(curl -fsS --max-time 5 -H "Metadata-Flavor: Google" \
      "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}" 2>/dev/null)"; then
    printf '%s' "${val}"
  else
    printf '%s' "${default_val}"
  fi
}

# --- 1. Prevent duplicate supervisor if startup script is re-triggered live ---
if [ -f "${SUPERVISOR_PIDFILE}" ]; then
  OLD_PID="$(cat "${SUPERVISOR_PIDFILE}" 2>/dev/null || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    log "Supervisor already running (pid=${OLD_PID}); exiting cleanly."
    STARTUP_OK=1
    exit 0
  fi
  rm -f "${SUPERVISOR_PIDFILE}"
fi

# --- 2. Read instance metadata ------------------------------------------------
DEFAULT_WORKERS="$(( $(nproc) / 2 ))"
if [ "${DEFAULT_WORKERS}" -lt 1 ]; then
  DEFAULT_WORKERS=1
fi

BUCKET="$(get_meta "bucket" "")"
CAMPAIGN_ID="$(get_meta "campaign-id" "revision130")"
BRANCH="$(get_meta "branch" "revision/schema-1.3.0")"
MAX_WORKERS="$(get_meta "max-workers" "${DEFAULT_WORKERS}")"
PRESET="$(get_meta "preset" "revision130")"
TIER="$(get_meta "tier" "all")"
SHARD="$(get_meta "shard" "")"
STAGGER="$(get_meta "stagger" "15")"
TOTAL_TIMESTEPS="$(get_meta "total-timesteps" "")"
EXTRA_ARGS="$(get_meta "extra-args" "--no-libsumo")"
SYNC_INTERVAL_S="$(get_meta "sync-interval-s" "1800")"
AUTO_SHUTDOWN="$(get_meta "auto-shutdown" "true")"

if [ -n "${SHARD}" ]; then
  SHARD_I="${SHARD%%/*}"
  SHARD_N="${SHARD##*/}"
  if [ -z "${SHARD_I}" ] || [ -z "${SHARD_N}" ] || [ "${SHARD_I}" = "${SHARD}" ]; then
    log "ERROR: invalid shard metadata '${SHARD}' (expected i/n, e.g. 1/4)."
    exit 1
  fi
  SHARD_TAG="shard${SHARD_I}of${SHARD_N}"
  MANIFEST_NAME="campaign_manifest_${SHARD_TAG}.json"
  GCS_STARTUP_LOG="startup_${SHARD_TAG}.log"
  GCS_SUPERVISOR_LOG="supervisor_${SHARD_TAG}.log"
  GCS_LAUNCHER_LOG="launcher_${SHARD_TAG}.log"
fi

if [ -z "${BUCKET}" ]; then
  log "ERROR: instance metadata attribute 'bucket' is required (e.g. --metadata=bucket=my-bucket)."
  exit 1
fi
if [ -z "${CAMPAIGN_ID}" ]; then
  log "ERROR: instance metadata attribute 'campaign-id' must not be empty."
  exit 1
fi

log "Config: branch=${BRANCH} campaign_id=${CAMPAIGN_ID} bucket=gs://${BUCKET}/${CAMPAIGN_ID}/ max_workers=${MAX_WORKERS} preset=${PRESET} tier=${TIER} shard=${SHARD:-none} manifest=${MANIFEST_NAME}"

# --- 3. OS packages + gsutil (idempotent) ------------------------------------
if ! command -v git >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1 || ! dpkg -s python3-venv >/dev/null 2>&1; then
  log "Installing base OS packages..."
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    git curl ca-certificates gnupg apt-transport-https \
    python3 python3-venv python3-pip libglib2.0-0
fi

if ! command -v gsutil >/dev/null 2>&1; then
  log "gsutil not found on PATH (including /snap/bin); installing google-cloud-cli from Google apt repo..."
  apt-get update -qq
  apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl
  install -d -m 0755 /usr/share/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -qq
  apt-get install -y --no-install-recommends google-cloud-cli
fi

if ! command -v gsutil >/dev/null 2>&1; then
  log "ERROR: gsutil still not found in PATH after google-cloud-cli installation."
  exit 1
fi

# --- 4. Clone or update public repository ------------------------------------
if [ -d "${REPO_DIR}/.git" ]; then
  log "Repo exists at ${REPO_DIR}; fetching branch ${BRANCH}..."
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}" || true
  git -C "${REPO_DIR}" checkout "${BRANCH}" || git -C "${REPO_DIR}" checkout -B "${BRANCH}" "origin/${BRANCH}" || true
  git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}" || true
else
  log "Cloning ${REPO_URL} (branch=${BRANCH}) into ${REPO_DIR}..."
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
fi

# --- 5. Python venv + minimal pinned training/eval deps + SUMO 1.24.0 --------
if [ ! -f "${DEPS_MARKER}" ] || [ ! -x "${VENV_DIR}/bin/sumo" ]; then
  log "Setting up Python venv at ${VENV_DIR}..."
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3 -m venv "${VENV_DIR}"
  fi
  "${VENV_DIR}/bin/pip" install --upgrade pip wheel

  # Install ONLY what training + eval + paper artifact scripts import
  # (no ultralytics, opencv-python, streamlit, pyserial, torchvision, or CUDA).
  # Pinned to the tested local Python 3.10 environment where possible.
  log "Installing CPU-only PyTorch..."
  "${VENV_DIR}/bin/pip" install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.11.0"

  log "Installing minimal pinned training/eval dependencies..."
  "${VENV_DIR}/bin/pip" install \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "pandas==2.2.3" \
    "matplotlib==3.10.3" \
    "gymnasium==1.2.3" \
    "pettingzoo==1.25.0" \
    "tensorboard==2.13.0"

  # Pin eclipse-sumo + traci + sumolib to exact local SUMO version (1.24.0).
  # Note: libsumo is intentionally not installed so train_ppo uses stable traci subprocesses.
  log "Installing pinned eclipse-sumo==${SUMO_VERSION} traci==${SUMO_VERSION} sumolib==${SUMO_VERSION}..."
  "${VENV_DIR}/bin/pip" install --force-reinstall \
    "eclipse-sumo==${SUMO_VERSION}" \
    "traci==${SUMO_VERSION}" \
    "sumolib==${SUMO_VERSION}"

  # Capture sumo --version into a variable instead of piping to head under pipefail.
  SUMO_VER_OUT="$("${VENV_DIR}/bin/sumo" --version 2>&1)"
  SUMO_VER_LINE="${SUMO_VER_OUT%%$'\n'*}"
  log "Verified SUMO binary: ${SUMO_VER_LINE}"
  touch "${DEPS_MARKER}"
else
  log "Python venv and minimal deps + SUMO ${SUMO_VERSION} already installed (marker present)."
fi

SUMO_HOME="$("${VENV_DIR}/bin/python" -c "import sumo; print(sumo.SUMO_HOME)")"
log "Resolved SUMO_HOME=${SUMO_HOME}"

# Idempotent fallback: if a fresh clone is missing any canonical demand files
# (e.g. before untracked files are committed), generate them with venv duarouter.
# Already-present tracked files are never touched.
for net in n2_corridor n3_grid n4_irregular; do
  NET_DIR="${REPO_DIR}/sumo_configs/networks/${net}"
  if [ -d "${NET_DIR}" ] && { [ ! -f "${NET_DIR}/demand_train.rou.xml" ] || [ ! -f "${NET_DIR}/demand_eval.rou.xml" ]; }; then
    log "Missing demand files in ${net}; generating deterministically with venv duarouter..."
    (
      cd "${REPO_DIR}"
      PATH="${VENV_DIR}/bin:${PATH}" SUMO_HOME="${SUMO_HOME}" \
        "${VENV_DIR}/bin/python" "${REPO_DIR}/scripts/generate_demand.py" --network "${net}" --seed 42
    )
  fi
done

# Ensure sync source directories exist
mkdir -p "${REPO_DIR}/results" "${REPO_DIR}/models"

# Fresh-disk recovery: if a brand-new disk boots with an existing GCS campaign,
# pull prior results/ and models/ before launching. Uses the shard-specific
# manifest when SHARD is set so sharded VMs resume their own state cleanly.
LOCAL_CAMPAIGN_DIR="${REPO_DIR}/results/paper1_mappo/${CAMPAIGN_ID}"
if [ ! -d "${LOCAL_CAMPAIGN_DIR}" ]; then
  if gsutil -q stat "gs://${BUCKET}/${CAMPAIGN_ID}/results/paper1_mappo/${CAMPAIGN_ID}/${MANIFEST_NAME}" 2>/dev/null; then
    log "Fresh disk detected with existing GCS state (${MANIFEST_NAME}); restoring from gs://${BUCKET}/${CAMPAIGN_ID}/..."
    gsutil -m rsync -r "gs://${BUCKET}/${CAMPAIGN_ID}/results" "${REPO_DIR}/results" || true
    gsutil -m rsync -r "gs://${BUCKET}/${CAMPAIGN_ID}/models" "${REPO_DIR}/models" || true
  fi
fi

# --- 6. Write campaign env + persistent supervisor script --------------------
{
  printf 'REPO_DIR=%q\n' "${REPO_DIR}"
  printf 'VENV_DIR=%q\n' "${VENV_DIR}"
  printf 'SUMO_HOME=%q\n' "${SUMO_HOME}"
  printf 'BUCKET=%q\n' "${BUCKET}"
  printf 'CAMPAIGN_ID=%q\n' "${CAMPAIGN_ID}"
  printf 'MAX_WORKERS=%q\n' "${MAX_WORKERS}"
  printf 'PRESET=%q\n' "${PRESET}"
  printf 'TIER=%q\n' "${TIER}"
  printf 'SHARD=%q\n' "${SHARD}"
  printf 'SHARD_TAG=%q\n' "${SHARD_TAG}"
  printf 'MANIFEST_NAME=%q\n' "${MANIFEST_NAME}"
  printf 'GCS_STARTUP_LOG=%q\n' "${GCS_STARTUP_LOG}"
  printf 'GCS_SUPERVISOR_LOG=%q\n' "${GCS_SUPERVISOR_LOG}"
  printf 'GCS_LAUNCHER_LOG=%q\n' "${GCS_LAUNCHER_LOG}"
  printf 'STAGGER=%q\n' "${STAGGER}"
  printf 'TOTAL_TIMESTEPS=%q\n' "${TOTAL_TIMESTEPS}"
  printf 'EXTRA_ARGS=%q\n' "${EXTRA_ARGS}"
  printf 'SYNC_INTERVAL_S=%q\n' "${SYNC_INTERVAL_S}"
  printf 'AUTO_SHUTDOWN=%q\n' "${AUTO_SHUTDOWN}"
  printf 'LOG_DIR=%q\n' "${LOG_DIR}"
  printf 'STARTUP_LOG=%q\n' "${STARTUP_LOG}"
  printf 'SUPERVISOR_LOG=%q\n' "${SUPERVISOR_LOG}"
  printf 'SUPERVISOR_PIDFILE=%q\n' "${SUPERVISOR_PIDFILE}"
} > "${ENV_FILE}"
chmod 0644 "${ENV_FILE}"

cat <<'EOF' > "${SUPERVISOR_SCRIPT}"
#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source /etc/sim2real-campaign.env

export PATH="${VENV_DIR}/bin:/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
export SUMO_HOME
export PYTHONUNBUFFERED=1

if [ -n "${SHARD_TAG:-}" ]; then
  LAUNCHER_LOG="${LOG_DIR}/launcher_${CAMPAIGN_ID}_${SHARD_TAG}.log"
else
  LAUNCHER_LOG="${LOG_DIR}/launcher_${CAMPAIGN_ID}.log"
fi
MANIFEST_PATH="${REPO_DIR}/results/paper1_mappo/${CAMPAIGN_ID}/${MANIFEST_NAME}"
LAUNCHER_PID=""
PREEMPTED=0

log() {
  printf '[%s] [supervisor] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

sync_to_gcs() {
  local rc=0
  log "Syncing results/ and models/ -> gs://${BUCKET}/${CAMPAIGN_ID}/ ..."
  if [ -d "${REPO_DIR}/results" ]; then
    gsutil -m rsync -r "${REPO_DIR}/results" "gs://${BUCKET}/${CAMPAIGN_ID}/results" || rc=$?
  fi
  if [ -d "${REPO_DIR}/models" ]; then
    gsutil -m rsync -r "${REPO_DIR}/models" "gs://${BUCKET}/${CAMPAIGN_ID}/models" || rc=$?
  fi
  if [ -f "${LAUNCHER_LOG}" ]; then
    gsutil cp "${LAUNCHER_LOG}" "gs://${BUCKET}/${CAMPAIGN_ID}/${GCS_LAUNCHER_LOG}" || true
  fi
  if [ -f "${SUPERVISOR_LOG}" ]; then
    gsutil cp "${SUPERVISOR_LOG}" "gs://${BUCKET}/${CAMPAIGN_ID}/${GCS_SUPERVISOR_LOG}" || true
  fi
  if [ -f "${STARTUP_LOG}" ]; then
    gsutil cp "${STARTUP_LOG}" "gs://${BUCKET}/${CAMPAIGN_ID}/${GCS_STARTUP_LOG}" || true
  fi
  log "Sync finished (rc=${rc})."
  return "${rc}"
}

count_completed_jobs() {
  if [ ! -f "${MANIFEST_PATH}" ]; then
    printf '0'
    return 0
  fi
  "${VENV_DIR}/bin/python" - "${MANIFEST_PATH}" <<'PYEOF' 2>/dev/null || printf '0'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    counts = data.get("counts") or {}
    print(int(counts.get("done", 0)) + int(counts.get("skipped", 0)))
except Exception:
    print(0)
PYEOF
}

on_term() {
  PREEMPTED=1
  log "Received termination signal (likely Spot preemption); stopping launcher cleanly..."
  if [ -n "${LAUNCHER_PID}" ] && kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    kill -TERM "${LAUNCHER_PID}" 2>/dev/null || true
    # Wait up to 25s (launcher child-termination deadline is <=10s; GCP gives ~30s).
    for _ in $(seq 1 25); do
      if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  sync_to_gcs || true
  rm -f "${SUPERVISOR_PIDFILE}"
  exit 0
}

trap on_term TERM INT

echo $$ > "${SUPERVISOR_PIDFILE}"
cd "${REPO_DIR}"

CMD=(
  "${VENV_DIR}/bin/python"
  "${REPO_DIR}/scripts/parallel_launcher.py"
  "--campaign-id" "${CAMPAIGN_ID}"
  "--max-workers" "${MAX_WORKERS}"
  "--stagger" "${STAGGER}"
)
if [ -n "${PRESET}" ]; then
  CMD+=("--preset" "${PRESET}")
fi
if [ -n "${TIER}" ]; then
  CMD+=("--tier" "${TIER}")
fi
if [ -n "${SHARD}" ]; then
  CMD+=("--shard" "${SHARD}")
fi
if [ -n "${TOTAL_TIMESTEPS}" ]; then
  CMD+=("--total-timesteps" "${TOTAL_TIMESTEPS}")
fi
if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( ${EXTRA_ARGS} )
  CMD+=("--" "${EXTRA_ARR[@]}")
fi

log "Starting launcher: ${CMD[*]}"
nohup "${CMD[@]}" </dev/null >>"${LAUNCHER_LOG}" 2>&1 &
LAUNCHER_PID=$!
log "Launcher started (pid=${LAUNCHER_PID}), log=${LAUNCHER_LOG}, manifest=${MANIFEST_PATH}"

ELAPSED=0
TICK=30
while kill -0 "${LAUNCHER_PID}" 2>/dev/null; do
  sleep "${TICK}" &
  wait $! || true
  if [ "${PREEMPTED}" -eq 1 ]; then
    exit 0
  fi
  ELAPSED=$(( ELAPSED + TICK ))
  if [ "${ELAPSED}" -ge "${SYNC_INTERVAL_S}" ]; then
    ELAPSED=0
    sync_to_gcs || true
  fi
done

LAUNCHER_RC=0
wait "${LAUNCHER_PID}" || LAUNCHER_RC=$?
COMPLETED_JOBS="$(count_completed_jobs)"
log "Launcher exited with status ${LAUNCHER_RC} (completed_jobs=${COMPLETED_JOBS})."

# Final sync (retry once on transient error)
if ! sync_to_gcs; then
  log "Final sync failed; retrying in 15s..."
  sleep 15
  sync_to_gcs || true
fi

rm -f "${SUPERVISOR_PIDFILE}"

# Guard against crash-looping: if launcher exited non-zero with zero completed
# jobs, do NOT retry — logs have been synced, shut down cleanly.
if [ "${LAUNCHER_RC}" -ne 0 ] && [ "${COMPLETED_JOBS}" -eq 0 ]; then
  log "ERROR: launcher exited non-zero (rc=${LAUNCHER_RC}) with 0 completed jobs; synced logs and stopping (no crash-loop)."
  if [ "${AUTO_SHUTDOWN}" = "true" ]; then
    shutdown -h now || true
  fi
  exit "${LAUNCHER_RC}"
fi

if [ "${AUTO_SHUTDOWN}" = "true" ]; then
  log "Campaign complete (rc=${LAUNCHER_RC}, completed_jobs=${COMPLETED_JOBS}); shutting down VM now."
  shutdown -h now || true
else
  log "Campaign complete (rc=${LAUNCHER_RC}, completed_jobs=${COMPLETED_JOBS}); auto-shutdown disabled."
fi
EOF
chmod 0755 "${SUPERVISOR_SCRIPT}"

# --- 7. Launch supervisor under nohup ----------------------------------------
log "Launching supervisor in background under nohup..."
nohup "${SUPERVISOR_SCRIPT}" </dev/null >>"${SUPERVISOR_LOG}" 2>&1 &
SUP_PID=$!
sleep 2
if kill -0 "${SUP_PID}" 2>/dev/null; then
  log "Supervisor running (pid=${SUP_PID}). Tail ${SUPERVISOR_LOG} or ${LOG_DIR}/launcher_${CAMPAIGN_ID}*.log to monitor."
  STARTUP_OK=1
else
  SUP_RC=0
  wait "${SUP_PID}" || SUP_RC=$?
  if [ "${SUP_RC}" -eq 0 ]; then
    log "Supervisor exited cleanly (pid=${SUP_PID}, rc=0); check ${SUPERVISOR_LOG}."
    STARTUP_OK=1
  else
    log "ERROR: supervisor pid=${SUP_PID} exited quickly with rc=${SUP_RC}; check ${SUPERVISOR_LOG}."
    exit "${SUP_RC}"
  fi
fi
