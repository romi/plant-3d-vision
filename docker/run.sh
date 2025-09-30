#!/bin/bash

# --------------------------------
# Functions for colors and messages
# --------------------------------
setup_colors() {
  RED="\033[0;31m"    # Define red color code
  GREEN="\033[0;32m"  # Define green color code
  YELLOW="\033[0;33m" # Define yellow color code
  BLUE="\033[0;34m"   # Define blue color code for debug messages
  NC="\033[0m"        # No Color code to reset colors
  INFO="${GREEN}INFO${NC}    "    # Prefix for info messages
  WARNING="${YELLOW}WARNING${NC} " # Prefix for warning messages
  ERROR="${RED}$(bold ERROR)${NC}   " # Prefix for error messages using bold function
  DEBUG="${BLUE}DEBUG${NC}   "   # Prefix for debug messages
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

log_debug() {
  if [ "${DEBUG_MODE}" = true ]; then
    echo -e "${DEBUG}$1" # Print debug message with DEBUG prefix if debug mode is enabled
  fi
}

# --------------------------------
# Functions for script initialization
# --------------------------------
initialize_variables() {
  # Default group id to use when starting the container:
  gid=2020
  # Docker image tag to use, 'latest' by default:
  VTAG="latest"
  # Command to execute after starting the docker container:
  cmd=''
  # Volume mounting options:
  mount_option=""
  # Self-test flag (0/1 to indicate call to a test)
  SELF_TEST=0
  # Debug mode is disabled by default
  DEBUG_MODE=false

  # Define test commands
  unittest_cmd="python3 -m unittest discover -s plant-3d-vision/tests/unit/"
  integration_test_cmd="python3 -m unittest discover -s plant-3d-vision/tests/integration/"
  pipeline_cmd="cd plant-3d-vision/ && ./tests/check_pipe.sh"
  geom_pipeline_cmd="cd plant-3d-vision/ && ./tests/check_geom_pipe.sh"
  ml_pipeline_cmd="cd plant-3d-vision/ && ./tests/check_ml_pipe.sh"
  gpu_cmd="nvidia-smi"
  webterm_cmd="gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8080 plant3dvision.webterm.wsgi:application"

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
    "By default, use the '${VTAG}' tag."
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
  echo "  --webterm
    Starts the WebTerm application." \
    "It will bind the host port 8080 to the container port 8080."
  # -- Debug option:
  echo "  --debug
    Enable debug mode to print additional debug information."
  # -- General options:
  echo "  -h, --help
    Output a usage message and exit."

  echo ""

  echo "$(bold TEST OPTIONS):"
  echo "You may select ONE of the test option below to execute this test instead of accessing the terminal or running a command."
  echo "  --unittest
    Run the unit tests defined in 'plant-3d-vision/tests/unit'."
  echo "  --test-integration
    Run the integration tests defined in 'plant-3d-vision/tests/integration'."
  echo "  --test-pipelines
    Run the reconstruction & quantification pipelines (geometric & machine-learning based) on the 'real_plant test dataset."
  echo "  --test-geom-pipeline
    Run the reconstruction & quantification pipeline using the geometric based workflow on the 'real_plant test dataset." \
    "Test dataset are located under 'tests/testdata'."
  echo "  --test-ml-pipeline
    Run the reconstruction & quantification pipeline using the machine-learning based workflow on the 'real_plant test dataset." \
    "Test dataset are located under 'tests/testdata'."
  echo "  --test-gpu
    Test correct access to NVIDIA GPU resources from docker container."
}

# --------------------------------
# Database setup functions
# --------------------------------
check_database_environment() {
  if [ -z ${ROMI_DB+x} ] && [ ${SELF_TEST} -eq 0 ]; then
    log_warning "Environment variable 'ROMI_DB' is not defined, set it to use as default database location!"
  fi
}

setup_database_mount() {
  if [ -n "${host_db}" ]; then
    mount_option="${mount_option} -v ${host_db}:/myapp/db"
    log_info "Automatic bind mount of '${host_db}' (host) to '/myapp/db' (container)!"
  else
    # Only raise ERROR message if not a SELF-TEST:
    if [ ${SELF_TEST} -eq 0 ]; then
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
      if [ ${SELF_TEST} -eq 0 ]; then
        log_error "Group name for host database '${host_db}' could not be retrieved!"
        exit 1
      fi
    fi
  else
    # Only raise WARNING message if not a SELF-TEST:
    if [ ${SELF_TEST} -eq 0 ]; then
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
      VTAG=$1
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
      SELF_TEST=1
      log_info "Running unitary tests..."
      ;;
    --test-integration)
      cmd=${integration_test_cmd}
      SELF_TEST=1
      log_info "Running integration tests..."
      ;;
    --test-pipelines)
      cmd=${pipeline_cmd}
      SELF_TEST=1
      log_info "Running reconstruction pipeline self-tests (geometric & machine-learning based)..."
      ;;
    --test-geom-pipeline)
      cmd=${geom_pipeline_cmd}
      SELF_TEST=1
      log_info "Running reconstruction pipeline self-test using geometric based workflow..."
      ;;
    --test-ml-pipeline)
      cmd=${ml_pipeline_cmd}
      SELF_TEST=1
      log_info "Running reconstruction pipeline self-test using machine-learning based workflow..."
      ;;
    --test-gpu)
      cmd=${gpu_cmd}
      SELF_TEST=1
      log_info "Running GPU self-test procedure..."
      ;;
    --webterm)
      cmd=${webterm_cmd}
      docker_option="${docker_option} -p 8080:8080"
      log_info "Starting WebTerm..."
      ;;
    -v | --volume)
      shift
      mount_option="${mount_option} -v $1"
      ;;
    --debug)
      DEBUG_MODE=true
      log_debug "Debug mode enabled"
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
  # Construct the docker run command
  docker_cmd="docker run --rm --gpus all"
  docker_cmd+=" ${mount_option}"
  docker_cmd+=" --user romi:${gid}"
  docker_cmd+=" ${docker_option}"
  docker_cmd+=" -i"  # use the `-i` flag to load `~/.bashrc`.
  docker_cmd+=" ${USE_TTY}"
  docker_cmd+=" roboticsmicrofarms/plant-3d-vision:${VTAG}"
  docker_cmd+=" bash"

  # Print the run configuration options
  log_debug "Run configuration:"
  log_debug "- Docker image: roboticsmicrofarms/plant-3d-vision:${VTAG}"
  log_debug "- Docker bind mount: ${mount_option}"
  log_debug "- Docker options: ${docker_option}"
  # Print the full command that will be executed
  log_debug "Executing command: ${docker_cmd}"

  # Execute the docker run command
  eval ${docker_cmd}
}

run_docker_command() {
  # Construct the docker run command
  docker_cmd="docker run --rm --gpus all"
  docker_cmd+=" ${mount_option}"
  docker_cmd+=" --user romi:${gid}"
  docker_cmd+=" ${docker_option}"
  docker_cmd+=" -i"  # use the `-i` flag to load `~/.bashrc`.
  docker_cmd+=" ${USE_TTY}"
  docker_cmd+=" roboticsmicrofarms/plant-3d-vision:${VTAG}"
  docker_cmd+=" \"${cmd}\""

  # Print the run configuration options
  log_debug "Run configuration:"
  log_debug "- Docker image: roboticsmicrofarms/plant-3d-vision:${VTAG}"
  log_debug "- Docker bind mount: ${mount_option}"
  log_debug "- Docker options: ${docker_option}"
  log_debug "- Command: ${cmd}"
  # Print the full command that will be executed
  log_debug "Executing command: ${docker_cmd}"

  # Get the date to estimate command execution time:
  start_time=$(date +%s)

  # Execute the docker run command
  eval ${docker_cmd}

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