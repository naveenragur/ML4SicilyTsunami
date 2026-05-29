#!/bin/bash
#SBATCH --account=PR_carisma
#SBATCH --job-name=oq_emul_risk
#SBATCH --time=0-24:00
#SBATCH --array=0-17%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpuq
#SBATCH --nodelist=cn08
#SBATCH --output=/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/jobs/log/slurm-oq-emul-%A_%a.out
#SBATCH --error=/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/jobs/log/slurm-oq-emul-%A_%a.err

set -eo pipefail

begin=$(date +%s)
echo "node: ${HOSTNAME}"
echo "start time: $(date)"
echo "task id: ${SLURM_ARRAY_TASK_ID:-0}"
echo "..........."

source "/home/${USER}/.bash_profile"

RISK_DIR="/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk"
JOBS_DIR="${RISK_DIR}/jobs"
HAZARD_DIR="${RISK_DIR}/hazard/oq_inputs"
TMP_DIR="${RISK_DIR}/tmp"
TEMPLATE_INI="${JOBS_DIR}/job_hdf5_emulation_template.ini"
VENV_ACTIVATE="/home/naveen.raguramalinga/opt/src/.venv/bin/activate"
OQ_BIN="${OQ_BIN:-/home/naveen.raguramalinga/opt/src/.venv/bin/oq}"
MAX_RUNTIME_MARKER="tsunami_portfolio_curves_stats.csv"
CLEAN_MARKER="${JOBS_DIR}/log/oq-clean-${SLURM_ARRAY_JOB_ID:-manual}.done"

sizes=(225 529 892 1658 3454 7071)
variants=(mean sigmaminus sigmaplus)

task_id=${SLURM_ARRAY_TASK_ID:-0}
num_variants=${#variants[@]}
size=${sizes[$((task_id / num_variants))]}
variant=${variants[$((task_id % num_variants))]}
token="${size}_${variant}"

gmfs_file="${HAZARD_DIR}/tsunami_prob_${token}.hdf5"
outdir="${TMP_DIR}/${size}/${variant}"
job_ini="${JOBS_DIR}/job_hdf5_${token}.ini"
rel_gmfs_file="../hazard/oq_inputs/tsunami_prob_${token}.hdf5"
rel_outdir="../tmp/${size}/${variant}"

if [[ ! -f "${gmfs_file}" ]]; then
  echo "Missing GMFs file: ${gmfs_file}" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE_INI}" ]]; then
  echo "Missing template INI: ${TEMPLATE_INI}" >&2
  exit 1
fi

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Missing OQ virtualenv activation script: ${VENV_ACTIVATE}" >&2
  exit 1
fi

source "${VENV_ACTIVATE}"
set -u
export OQ_DISTRIBUTE=processpool
echo "OQ_DISTRIBUTE: ${OQ_DISTRIBUTE}"

if [[ "${CLEAN_OQ_QUEUE:-0}" == "1" ]]; then
  if [[ "${task_id}" == "0" ]]; then
    echo "Cleaning stale uncompleted OpenQuake calculations before starting the array"
    "${OQ_BIN}" engine --yes --delete-uncompleted-calculations
    touch "${CLEAN_MARKER}"
  else
    echo "Waiting for OpenQuake cleanup marker: ${CLEAN_MARKER}"
    for _ in {1..120}; do
      [[ -f "${CLEAN_MARKER}" ]] && break
      sleep 5
    done
    if [[ ! -f "${CLEAN_MARKER}" ]]; then
      echo "Timed out waiting for ${CLEAN_MARKER}" >&2
      exit 1
    fi
  fi
fi

mkdir -p "${outdir}"

if [[ -f "${outdir}/${MAX_RUNTIME_MARKER}" && "${FORCE:-0}" != "1" ]]; then
  echo "Skipping ${token}: ${MAX_RUNTIME_MARKER} already exists in ${outdir}"
  exit 0
fi

sed \
  -e "s|^description = .*|description = Native OQ event-based risk using FLOWDEPTH imported GMFs (${token}) and tsunami.main AAL/EP integration|" \
  -e "s|^exposure_file = .*|exposure_file = ../exposure/oq_inputs/exposure_model.xml|" \
  -e "s|^structural_vulnerability_file = .*|structural_vulnerability_file = ../vulnerability/oq_inputs/vulnerability_model_cm.xml|" \
  -e "s|^gmfs_file = .*|gmfs_file = ${rel_gmfs_file}|" \
  -e "s|^postrisk_args = .*|postrisk_args = {'event_rates_file': '../hazard/oq_inputs/event_rate_samples.csv','plot': True,'plot_loss_ratio': True}|" \
  -e "s|^export_dir = .*|export_dir = ${rel_outdir}|" \
  "${TEMPLATE_INI}" > "${job_ini}"

echo "Running ${token}"
echo "job ini: ${job_ini}"
echo "output: ${outdir}"

cd "${JOBS_DIR}"
"${OQ_BIN}" engine --run "${job_ini}"
"${OQ_BIN}" export risk_by_event -1 -d "${outdir}"

echo "..........."
end=$(date +%s)
elapsed=$((end - begin))
echo "Time taken: ${elapsed} seconds"
