#!/bin/bash

# --------------------------------
# Functions for colors and messages
# --------------------------------
setup_colors() {
  RED="\033[0;31m"    # Define red color code
  GREEN="\033[0;32m"  # Define green color code
  YELLOW="\033[0;33m" # Define yellow color code
  NC="\033[0m"        # No Color code to reset colors
  INFO="${GREEN}INFO${NC}    "    # Prefix for info messages
  WARNING="${YELLOW}WARNING${NC} " # Prefix for warning messages
  ERROR="${RED}$(bold ERROR)${NC}   " # Prefix for error messages using bold function
}

bold() {
  echo -e "\e[1m$*\e[0m" # Make text bold and reset
}

log_info() {
  echo -e "${INFO}$1" # Print info message with INFO prefix
}

log_warning() {
  echo -e "${WARNING}$1" # Print warning message with WARNING prefix
}

log_error() {
  echo -e "${ERROR}$1" # Print error message with ERROR prefix
}

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Default group id to use when starting the container:
  gid=2020
  # Docker image tag to use, 'latest' by default:
  vtag="latest"
  # Command to execute after starting the docker container:
  cmd=''
  # Volume mounting options:
  mount_option=""
  # Self-test flag (0/1 to indicate call to a test)
  self_test=0

  # Define test commands
  unittest_cmd="python3 -m unittest discover -s plant-3d-vision/tests/unit/"
  integration_test_cmd="python3 -m unittest discover -s plant-3d-vision/tests/integration/"
  pipeline_cmd="cd plant-3d-vision/ && ./tests/check_pipe.sh"
  geom_pipeline_cmd="cd plant-3d-vision/ && ./tests/check_geom_pipe.sh"
  ml_pipeline_cmd="cd plant-3d-vision/ && ./tests/check_ml_pipe.sh"
  gpu_cmd="nvidia-smi"

  # If the `ROMI_DB` variable is set, use it as the default database location; else set it to empty:
  if [ -z ${ROMI_DB+x} ]; then
    host_db=''
  else
    host_db=${ROMI_DB}
  fi
}

# --------------------------------
# Usage information function
# --------------------------------
show_usage() {
  echo -e "$(bold USAGE):"
  echo "  ./docker/run.sh [OPTIONS] [TEST OPTION]"
  echo ""

  echo -e "$(bold DESCRIPTION):"
  echo "  Start a docker container using the 'roboticsmicrofarms/plant-3d-vision' image.

  It must be run from the 'plant-3d-vision' repository root folder if you wish to use one of the self-testing option!"
  echo ""

  echo -e "$(bold OPTIONS):"
  echo "  -t, --tag
    Image tag to use." \
    "By default, use the '${vtag}' tag."
  echo "  -db, --database
    Path to the host database to mount inside the docker container." \
    "By default, use the 'ROMI_DB' environment variable (if defined)."
  echo "  -v, --volume
    Volume mapping between host and container to mount a local directory in the container." \
    "Absolute paths are required and multiple use of this option is allowed." \
    "For example '-v /host/dir:/container/dir' makes the '/host/dir' directory accessible under '/container/dir' within the container."
  echo "  -c, --cmd
    Defines the command to run at container startup." \
    "By default, start an interactive container with a bash shell."
  echo "  -h, --help
    Output a usage message and exit."
  echo ""

  echo "$(bold TEST OPTIONS):"
  echo "You may select ONE of the test option below to execute this test instead of accessing the terminal or running a command."
  echo "  --unittest
    Run the unit tests defined in 'plant-3d-vision/tests/unit'."
  echo "  --integration_test
    Run the integration tests defined in 'plant-3d-vision/tests/integration'."
  echo "  --pipeline_test
    Run the reconstruction & quantification pipelines (geometric & machine-learning based) on the 'real_plant test dataset."
  echo "  --geom_pipeline_test
    Run the reconstruction & quantification pipeline using the geometric based workflow on the 'real_plant test dataset." \
    "Test dataset are located under 'tests/testdata'."
  echo "  --ml_pipeline_test
    Run the reconstruction & quantification pipeline using the machine-learning based workflow on the 'real_plant test dataset." \
    "Test dataset are located under 'tests/testdata'."
  echo "  --gpu_test
    Test correct access to NVIDIA GPU resources from docker container."
}

# --------------------------------
# Database setup functions
# --------------------------------
check_database_environment() {
  if [ -z ${ROMI_DB+x} ] && [ ${self_test} -eq 0 ]; then
    log_warning "Environment variable 'ROMI_DB' is not defined, set it to use as default database location!"
  fi
}

setup_database_mount() {
  if [ -n "${host_db}" ]; then
    mount_option="${mount_option} -v ${host_db}:/myapp/db"
    log_info "Automatic bind mount of '${host_db}' (host) to '/myapp/db' (container)!"
  else
    # Only raise ERROR message if not a SELF-TEST:
    if [ ${self_test} -eq 0 ]; then
      log_error "No local host database defined!"
      log_info "Set 'ROMI_DB' or use the '-db' | '--database' option to define it."
      exit 1
    fi
  fi
}

setup_user_group() {
  if [ -n "${host_db}" ]; then
    group_name=$(stat -c "%G" "${host_db}")
    if [ -n "${group_name}" ]; then
      gid=$(getent group "${group_name}" | cut --delimiter ':' --fields 3)
      log_info "Using host database path group name '${group_name}' & '${gid}'."
    else
      # Only raise next ERROR message if not a SELF-TEST:
      if [ ${self_test} -eq 0 ]; then
        log_error "Group name for host database '${host_db}' could not be retrieved!"
        exit 1
      fi
    fi
  else
    # Only raise WARNING message if not a SELF-TEST:
    if [ ${self_test} -eq 0 ]; then
      log_warning "Using default group id '${gid}'."
    fi
  fi
}

# --------------------------------
# Command line parsing function
# --------------------------------
parse_arguments() {
  docker_option=""
  while [ "$1" != "" ]; do
    case $1 in
    -t | --tag)
      shift
      vtag=$1
      ;;
    -db | --database)
      shift
      host_db=$1
      log_info "Got a manually defined database: ${host_db}"
      ;;
    -c | --cmd)
      shift
      cmd=$1
      ;;
    --unittest)
      cmd=${unittest_cmd}
      self_test=1
      log_info "Running unitary tests..."
      ;;
    --integration_test)
      cmd=${integration_test_cmd}
      self_test=1
      log_info "Running integration tests..."
      ;;
    --pipeline_test)
      cmd=${pipeline_cmd}
      self_test=1
      log_info "Running reconstruction pipeline self-tests (geometric & machine-learning based)..."
      ;;
    --geom_pipeline_test)
      cmd=${geom_pipeline_cmd}
      self_test=1
      log_info "Running reconstruction pipeline self-test using geometric based workflow..."
      ;;
    --ml_pipeline_test)
      cmd=${ml_pipeline_cmd}
      self_test=1
      log_info "Running reconstruction pipeline self-test using machine-learning based workflow..."
      ;;
    --gpu_test)
      cmd=${gpu_cmd}
      self_test=1
      log_info "Running GPU self-test procedure..."
      ;;
    -v | --volume)
      shift
      mount_option="${mount_option} -v $1"
      ;;
    -h | --help)
      show_usage
      exit 0
      ;;
    *)
      docker_option="${docker_option} $1"
      ;;
    esac
    shift
  done

  if [ "${docker_option}" != "" ]; then
    log_info "Extra docker arguments: '${docker_option}'!"
  fi
}

# --------------------------------
# Terminal handling function
# --------------------------------
check_terminal() {
  if [ -t 1 ]; then
    USE_TTY="-t"
  else
    USE_TTY=""
  fi
}

# --------------------------------
# Docker run functions
# --------------------------------
run_interactive_docker() {
  # Start in interactive mode, using the `-i` flag (load `~/.bashrc`).
  docker run --rm --gpus all ${mount_option} \
    --user romi:${gid} \
    --env PYOPENCL_CTX='0' \
    ${docker_option} \
    -i ${USE_TTY} \
    "roboticsmicrofarms/plant-3d-vision:${vtag}" \
    "bash"
}

run_docker_command() {
  log_info "Running: '${cmd}'."
  log_info "Bind mount: '${mount_option}'."

  # Get the date to estimate command execution time:
  start_time=$(date +%s)

  # Start in interactive mode, using the `-i` flag (load `~/.bashrc`).
  docker run --rm --gpus all ${mount_option} \
    --user romi:${gid} \
    --env PYOPENCL_CTX='0' \
    ${docker_option} \
    -i ${USE_TTY} \
    "roboticsmicrofarms/plant-3d-vision:${vtag}" \
    "${cmd}"

  # Get command exit code:
  cmd_status=$?

  # Print elapsed time if successful (code 0), else print command exit code
  elapsed_time=$(($(date +%s) - start_time))
  if [ ${cmd_status} -eq 0 ]; then
    log_info "Command SUCCEEDED in ${elapsed_time}s!"
  else
    log_error "Command FAILED after ${elapsed_time}s with code ${cmd_status}!"
  fi

  # Exit with status code:
  exit ${cmd_status}
}

# --------------------------------
# Main script execution
# --------------------------------
main() {
  setup_colors
  initialize_variables
  parse_arguments "$@"
  check_database_environment
  setup_database_mount
  setup_user_group
  check_terminal

  if [ "${cmd}" = "" ]; then
    run_interactive_docker
  else
    run_docker_command
  fi
}

# Execute main function with all arguments
main "$@"