#!/bin/bash

# -------------------------------------------------
# Color and logging helpers
# -------------------------------------------------
setup_colors() {
  GREEN="\033[0;32m"
  YELLOW="\033[0;33m"
  NC="\033[0m"                # No Color / reset
  INFO="${GREEN}INFO${NC}    "
  WARNING="${YELLOW}WARNING${NC} "
}

log_info() {
  echo -e "${INFO}$1"
}

log_warning() {
  echo -e "${WARNING}$1"
}

# -------------------------------------------------
# Main logic: download the model if missing
# -------------------------------------------------
main() {
  setup_colors

  model="Resnet_896_896_epoch50.pt"
  # Destination database directory can be overridden via the first argument
  db_path="${1:-tests/testdata}"

  if [[ -f "${db_path}/models/models/${model}" ]]; then
    log_info "Found trained CNN model file '${model}'."
  else
    log_warning "Could not find trained CNN model file '${model}'!"
    log_info "Downloading it via the 'setup_test_database' CLI..."
    setup_test_database real_plant --with-models --db-path "${db_path}"
  fi
}

# Execute
main "$@"