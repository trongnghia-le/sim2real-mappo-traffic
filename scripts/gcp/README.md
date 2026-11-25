# GCP Spot VM Campaign Runbook

Run the `revision130` training campaign (55 Tier-1 + 40 Tier-2 = 95 jobs, **300,000 timesteps per job** = 28.5M total env steps) on a single Google Cloud Spot VM with automatic in-place checkpoint resume across preemptions, 30-minute GCS sync, and auto-shutdown on completion.

## Sizing Recommendation

Measured peak RSS on `n3_grid` (2 concurrent jobs, `OMP_NUM_THREADS=1`, `--no-libsumo` / `traci`): **~537 MiB per job** (~508 MiB Python/PyTorch + ~29 MiB SUMO subprocess).
- 60 concurrent jobs ≈ **32 GiB** (~48–60 GiB with headroom for full 1080-step episodes, OS, and `gsutil`).
- Recommended machine family: **`highcpu`** (1 GB/vCPU), e.g. **`n2d-highcpu-128`** (128 vCPU, 128 GB RAM; ~4× headroom over measured RSS). A `standard` (4 GB/vCPU, 512 GB) machine is unnecessary.

## Variables

Set placeholders before running the commands below (run from the repo root):

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export BILLING_ACCOUNT_ID="<YOUR_BILLING_ACCOUNT_ID>"
export REGION="us-central1"
export ZONE="us-central1-a"
export BUCKET="<YOUR_UNIQUE_BUCKET_NAME>"
export VM_NAME="sim2real-mappo-spot"
export MACHINE_TYPE="n2d-highcpu-128"   # 128 vCPU, 128 GB RAM (highcpu)
export CAMPAIGN_ID="revision130"
export BRANCH="revision/schema-1.3.0"
export MAX_WORKERS="60"                 # default in startup.sh is nproc/2
export BUDGET_AMOUNT_USD="150"
```

## 1. Create the GCS Bucket

```bash
gcloud config set project "$PROJECT_ID"

gcloud storage buckets create "gs://${BUCKET}" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

## 2. Set a Budget Alert

```bash
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT_ID" \
  --display-name="sim2real-mappo-campaign-budget" \
  --budget-amount="${BUDGET_AMOUNT_USD}USD" \
  --filter-projects="projects/${PROJECT_ID}" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.8 \
  --threshold-rule=percent=1.0
```

## 3. Create the Spot VM

Uses `provisioning-model=SPOT` and `instance-termination-action=STOP` so the persistent disk (venv, repo, and in-progress checkpoints) survives preemption.

```bash
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write \
  --metadata="bucket=${BUCKET},campaign-id=${CAMPAIGN_ID},branch=${BRANCH},max-workers=${MAX_WORKERS},preset=revision130,tier=all" \
  --metadata-from-file="startup-script=scripts/gcp/startup.sh"
```

Optional metadata keys supported by `scripts/gcp/startup.sh`:
- `tier`: `1`, `2`, or `all` (default: `all`; Tier 1 always runs before Tier 2)
- `shard`: 1-indexed `i/n` partition (e.g. `1/2`, `2/2`) if splitting across VMs; writes per-shard manifest `campaign_manifest_shard{i}of{n}.json` and per-shard GCS logs (`launcher_shard{i}of{n}.log`, `supervisor_shard{i}of{n}.log`, `startup_shard{i}of{n}.log`)
- `stagger`: seconds between worker spawns (default: `15`)
- `total-timesteps`: override the `revision130` preset default of `300000` timesteps per job (non-preset matrix mode default: `2000000`)
- `extra-args`: flags passed to `train_ppo.py` after `--` (default: `--no-libsumo`)
- `auto-shutdown`: `true` (default) or `false` (also shuts down after best-effort log upload on startup failure or zero-progress launcher exit)

## 4. Watch Progress (and Resume After Preemption)

```bash
# Check VM status (RUNNING vs TERMINATED after preemption or auto-shutdown)
gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="value(status)"

# Tail startup, supervisor, or launcher logs over SSH
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="sudo tail -n 50 /var/log/sim2real-mappo/startup.log"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="sudo tail -f /var/log/sim2real-mappo/launcher_${CAMPAIGN_ID}.log"

# Inspect latest synced campaign manifest from GCS (updated every 30 min)
gsutil cat "gs://${BUCKET}/${CAMPAIGN_ID}/results/paper1_mappo/${CAMPAIGN_ID}/campaign_manifest.json" | head -n 30

# If preempted (status=TERMINATED before completion), restart to resume the same campaign-id:
gcloud compute instances start "$VM_NAME" --zone="$ZONE"
```

On restart, `startup.sh` skips completed jobs (presence of `bench.json`) and `train_ppo.py` auto-resumes interrupted jobs from their latest matching `last_model.pt`. When all jobs finish, `startup.sh` performs a final `gsutil -m rsync -r` and runs `shutdown -h now`.

## 5. Download Results and Clean Up

```bash
mkdir -p "./gcp_results/${CAMPAIGN_ID}"
gsutil -m rsync -r "gs://${BUCKET}/${CAMPAIGN_ID}/" "./gcp_results/${CAMPAIGN_ID}/"

# Delete the stopped VM once results are verified locally (avoids boot-disk storage charges)
gcloud compute instances delete "$VM_NAME" --zone="$ZONE"
```
