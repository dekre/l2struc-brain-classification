usage() {
  echo "usage: bash run_jupyter.sh  [ -p PORT ]    
    optional argument:
        -h,   [bool]      Show usage of otlrs.
        -p,   [string]    Specify port. Default port is 8888.        
    "
  exit 2
}

PORT=8888

VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

while [ "$1" != "" ]; do
  case "$1" in
  -h | --help)
    usage
    shift
    break
    ;;
  -p | --port)
    export PORT="$2"
    shift
    break
    ;;    
  # If invalid options were passed, then getopt should have reported an error,
  # which we checked as VALID_ARGUMENTS when getopt was called...
  *)    
    echo "Unknown command: $1 - this should not happen."
    usage
    exit 1 # error
    ;;
  esac
done

# Launch conda here
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Now start notebook in background
echo "Running jupyter on port ${PORT}"
conda activate base_fcst_engine
echo "Activating environment ${CONDA_DEFAULT_ENV}"
nohup jupyter lab --ip 0.0.0.0 --port ${PORT} --no-browser --debug > jupyter.log \
    & echo JPID=$!> jupyter.pid &