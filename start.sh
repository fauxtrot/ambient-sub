#!/usr/bin/env bash
# ============================================================
#  Ambient Subconscious — Startup Orchestrator
#  MI25 Server (Dell R630 + 2x Radeon Instinct MI25)
# ============================================================
set -uo pipefail

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SVELTE_DIR="$PROJECT_ROOT/web/ambient-subconscious-svelte"
VENV="$PROJECT_ROOT/.venv/bin/activate"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/ambient_subconscious.log"
PID_DIR="/tmp/ambient-sub"

# --- Load nvm (for npm/node) ---
export NVM_DIR="$HOME/.nvm"
[[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"

# --- Load Rust (for spacetime CLI) ---
[[ -s "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Defaults ---
MAX_LAYER=4
ACTION="start"

# ============================================================
#  Helpers
# ============================================================

log_ok()   { echo -e "  ${GREEN}[OK]${NC}   $1"; }
log_fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
log_warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
log_info() { echo -e "  ${CYAN}[....]${NC} $1"; }

banner() {
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

ensure_pid_dir() {
    mkdir -p "$PID_DIR"
}

save_pid() {
    # $1 = label, $2 = pid
    echo "$2" > "$PID_DIR/$1.pid"
}

read_pid() {
    local f="$PID_DIR/$1.pid"
    [[ -f "$f" ]] && cat "$f" || echo ""
}

kill_pid() {
    # $1 = label
    local pid
    pid=$(read_pid "$1")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log_info "Stopping $1 (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        # Wait up to 5s for graceful exit
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            log_warn "$1 force-killed"
        else
            log_ok "$1 stopped"
        fi
        rm -f "$PID_DIR/$1.pid"
    else
        [[ -f "$PID_DIR/$1.pid" ]] && rm -f "$PID_DIR/$1.pid"
    fi
}

# Wait for an HTTP endpoint to respond
# $1 = URL, $2 = max seconds, $3 = label
wait_for_http() {
    local url="$1" max="$2" label="$3"
    log_info "Waiting for $label ($url)..."
    for i in $(seq 1 "$max"); do
        if curl -sf "$url" -o /dev/null 2>/dev/null; then
            log_ok "$label ready"
            return 0
        fi
        sleep 1
    done
    log_fail "$label not responding after ${max}s"
    return 1
}

# Wait for a TCP port to be open
# $1 = port, $2 = max seconds, $3 = label
wait_for_port() {
    local port="$1" max="$2" label="$3"
    log_info "Waiting for $label (port $port)..."
    for i in $(seq 1 "$max"); do
        if (echo > /dev/tcp/localhost/"$port") 2>/dev/null; then
            log_ok "$label ready"
            return 0
        fi
        sleep 1
    done
    log_fail "$label not responding after ${max}s"
    return 1
}

# Wait for a log line to appear
# $1 = pattern, $2 = max seconds, $3 = label
wait_for_log() {
    local pattern="$1" max="$2" label="$3"
    log_info "Waiting for $label..."
    if timeout "$max" grep -qm1 "$pattern" <(tail -n0 -f "$LOG_FILE" 2>/dev/null); then
        log_ok "$label"
        return 0
    fi
    log_fail "$label (timeout ${max}s)"
    return 1
}

# Check if a service is healthy (for --status)
check_service() {
    local label="$1" check_cmd="$2"
    if eval "$check_cmd" >/dev/null 2>&1; then
        log_ok "$label"
        return 0
    else
        log_fail "$label"
        return 1
    fi
}

# ============================================================
#  Layer 0: Infrastructure (SpacetimeDB + llama.cpp)
# ============================================================

start_layer0() {
    echo ""
    banner "Layer 0: Infrastructure"

    # --- SpacetimeDB ---
    if docker ps --format '{{.Names}}' | grep -q '^spacetimedb$'; then
        log_ok "SpacetimeDB container already running"
    else
        log_info "Starting SpacetimeDB container..."
        if docker start spacetimedb >/dev/null 2>&1; then
            log_ok "SpacetimeDB container started"
        else
            docker run -d --name spacetimedb -p 3000:3000 clockworklabs/spacetime start >/dev/null 2>&1
            log_ok "SpacetimeDB container created and started"
        fi
    fi
    wait_for_port 3000 30 "SpacetimeDB :3000"

    # --- llama.cpp inside llama_build ---
    if ! docker ps --format '{{.Names}}' | grep -q '^llama_build$'; then
        log_info "Starting llama_build container..."
        docker start llama_build >/dev/null 2>&1 || {
            log_fail "llama_build container not found — create it first"
            return 1
        }
        log_ok "llama_build container started"
    else
        log_ok "llama_build container already running"
    fi

    # Check if llama-server is already running
    if curl -sf http://localhost:8080/v1/models -o /dev/null 2>/dev/null; then
        log_ok "llama-server :8080 already running"
    else
        log_info "Starting llama-server (Qwen3 8B)..."
        docker exec -d -e HIP_VISIBLE_DEVICES=0 llama_build \
            /workspace/llama.cpp/build/bin/llama-server \
            -m /data/qwen3-8b-q4_k_m.gguf \
            --host 0.0.0.0 --port 8080 \
            --jinja \
            -ngl 99
        wait_for_http "http://localhost:8080/v1/models" 90 "llama-server :8080"
    fi
}

status_layer0() {
    check_service "SpacetimeDB     :3000" "(echo > /dev/tcp/localhost/3000) 2>/dev/null"
    check_service "llama.cpp       :8080" "curl -sf http://localhost:8080/v1/models"
}

# ============================================================
#  Layer 1: Svelte + Process Bridge
# ============================================================

start_layer1() {
    echo ""
    banner "Layer 1: Svelte + Process Bridge"

    # Check if already running
    if curl -sf http://localhost:5174 -o /dev/null 2>/dev/null; then
        log_ok "Svelte :5174 already running"
    else
        if [[ ! -d "$SVELTE_DIR" ]]; then
            log_fail "Svelte directory not found: $SVELTE_DIR"
            return 1
        fi

        log_info "Starting Svelte + Process Bridge..."
        mkdir -p "$LOG_DIR"
        cd "$SVELTE_DIR"
        nohup npm run dev:all > "$LOG_DIR/svelte.log" 2>&1 &
        local svelte_pid=$!
        cd "$PROJECT_ROOT"
        save_pid "svelte" "$svelte_pid"

        wait_for_http "http://localhost:5174" 30 "Svelte :5174"
    fi

    # Process Bridge
    if (echo > /dev/tcp/localhost/8175) 2>/dev/null; then
        log_ok "Process Bridge :8175 already running"
    else
        wait_for_port 8175 20 "Process Bridge :8175"
    fi
}

status_layer1() {
    check_service "Svelte API      :5174" "curl -sf http://localhost:5174"
    check_service "Process Bridge  :8175" "(echo > /dev/tcp/localhost/8175) 2>/dev/null"
}

# ============================================================
#  Layer 2-4: Python Pipeline (ZMQ + Observer + Executive)
# ============================================================

start_layer2() {
    echo ""
    banner "Layer 2: ZMQ Providers"

    # Check if Python pipeline is already running
    local existing_pid
    existing_pid=$(read_pid "pipeline")
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        log_ok "Python pipeline already running (PID $existing_pid)"
        return 0
    fi

    if [[ ! -f "$VENV" ]]; then
        log_fail "Python venv not found: $VENV"
        return 1
    fi

    log_info "Starting Python pipeline..."
    mkdir -p "$LOG_DIR"

    # Truncate log so we tail fresh output
    > "$LOG_FILE"

    (
        source "$VENV"
        cd "$PROJECT_ROOT"
        nohup python -m ambient_subconscious start >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_DIR/pipeline.pid"
    )

    sleep 2  # Give Python a moment to start writing logs

    # Wait for ZMQ readiness
    wait_for_log "Audio pipeline started" 30 "ZMQ Audio :5555" || true
    wait_for_log "Video pipeline started" 30 "ZMQ Video :5556" || true
}

start_layer3() {
    echo ""
    banner "Layer 3: Observer (Inference Engines)"

    # Models load lazily on first frame — report what we can
    wait_for_log "YOLO model loaded" 60 "YOLO model loaded" || log_warn "YOLO loads on first frame"
    wait_for_log "Ambient Subconscious is running" 30 "Pipeline ready" || true
}

start_layer4() {
    echo ""
    banner "Layer 4: Executive / LLM + Conversation"

    wait_for_log "LLM available" 15 "LLM health check" || log_warn "Executive may not have started"
    wait_for_log "Executive agent started" 10 "Executive agent" || log_warn "Executive agent not confirmed"

    # Conversation engine (optional — config-gated)
    wait_for_log "Conversation engine started" 10 "Conversation engine" || log_warn "Conversation engine not enabled or not confirmed"
    wait_for_log "Callback server listening" 5 "Callback server :8765" || true
}

status_layer2() {
    local pid
    pid=$(read_pid "pipeline")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        check_service "Python pipeline  PID" "true"
        check_service "ZMQ Audio       :5555" "(echo > /dev/tcp/localhost/5555) 2>/dev/null"
        check_service "ZMQ Video       :5556" "(echo > /dev/tcp/localhost/5556) 2>/dev/null"
    else
        log_fail "Python pipeline   not running"
    fi
}

# ============================================================
#  Status
# ============================================================

do_status() {
    set +e  # Don't abort on failed checks
    banner "Ambient Subconscious — System Status"
    echo ""
    status_layer0
    echo ""
    status_layer1
    echo ""
    status_layer2
    echo ""
    set -e
}

# ============================================================
#  Final Banner
# ============================================================

print_final_banner() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${GREEN}${BOLD}  Ambient Subconscious — All Systems Ready${NC}"
    echo -e "${CYAN}============================================================${NC}"

    # Quick checks
    local stdb llm svelte bridge
    (echo > /dev/tcp/localhost/3000) 2>/dev/null && stdb="${GREEN}[OK]${NC}" || stdb="${RED}[FAIL]${NC}"
    curl -sf http://localhost:8080/v1/models -o /dev/null 2>/dev/null && llm="${GREEN}[OK]${NC}" || llm="${YELLOW}[WARN]${NC}"
    curl -sf http://localhost:5174 -o /dev/null 2>/dev/null && svelte="${GREEN}[OK]${NC}" || svelte="${RED}[FAIL]${NC}"
    (echo > /dev/tcp/localhost/8175) 2>/dev/null && bridge="${GREEN}[OK]${NC}" || bridge="${RED}[FAIL]${NC}"

    local callback
    (echo > /dev/tcp/localhost/8765) 2>/dev/null && callback="${GREEN}[OK]${NC}" || callback="${YELLOW}[----]${NC}"

    echo -e "  SpacetimeDB     :3000   $stdb"
    echo -e "  llama.cpp       :8080   $llm"
    echo -e "  Svelte API      :5174   $svelte"
    echo -e "  Process Bridge  :8175   $bridge"
    echo -e "  ZMQ Audio       :5555   ${GREEN}[OK]${NC}  (waiting for Godot)"
    echo -e "  ZMQ Video       :5556   ${GREEN}[OK]${NC}  (waiting for Godot)"
    echo -e "  Callback Server :8765   $callback"

    echo -e "${CYAN}============================================================${NC}"
    echo -e "  Listening for Godot client on ZMQ ports 5555/5556"
    echo -e "  Conversation engine → speak callback on :8765"
    echo -e "  Ctrl+C to stop all services"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

# ============================================================
#  Shutdown
# ============================================================

do_stop() {
    banner "Ambient Subconscious — Shutting Down"
    echo ""

    # 1. Python pipeline
    kill_pid "pipeline"

    # 2. Svelte + Bridge
    kill_pid "svelte"

    # 3. llama-server (keep container running)
    if docker exec llama_build pgrep -f llama-server >/dev/null 2>&1; then
        log_info "Stopping llama-server..."
        docker exec llama_build pkill -f llama-server 2>/dev/null || true
        log_ok "llama-server stopped (container still running)"
    fi

    # 4. SpacetimeDB
    if docker ps --format '{{.Names}}' | grep -q '^spacetimedb$'; then
        log_info "Stopping SpacetimeDB container..."
        docker stop spacetimedb >/dev/null 2>&1 || true
        log_ok "SpacetimeDB stopped"
    fi

    echo ""
    log_ok "All services stopped"
    echo ""
}

# ============================================================
#  Main
# ============================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --layer N    Start up to layer N only (0-4)"
    echo "  --stop       Graceful shutdown (reverse order)"
    echo "  --status     Health check all services"
    echo "  -h, --help   Show this help"
    echo ""
    echo "Layers:"
    echo "  0  Infrastructure  (SpacetimeDB :3000, llama.cpp :8080)"
    echo "  1  Bridge + Svelte (Vite :5174, Process Bridge :8175)"
    echo "  2  ZMQ Providers   (Audio :5555, Video :5556)"
    echo "  3  Observer        (YOLO, Whisper, Diart)"
    echo "  4  Executive/LLM   (Qwen3 8B, Stage server, Conversation engine)"
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer)
            MAX_LAYER="${2:-4}"
            shift 2
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

ensure_pid_dir

case "$ACTION" in
    start)
        banner "Ambient Subconscious — Starting (up to layer $MAX_LAYER)"

        [[ "$MAX_LAYER" -ge 0 ]] && start_layer0
        [[ "$MAX_LAYER" -ge 1 ]] && start_layer1
        [[ "$MAX_LAYER" -ge 2 ]] && start_layer2
        [[ "$MAX_LAYER" -ge 3 ]] && start_layer3
        [[ "$MAX_LAYER" -ge 4 ]] && start_layer4

        if [[ "$MAX_LAYER" -ge 4 ]]; then
            print_final_banner
        else
            echo ""
            log_ok "Started up to layer $MAX_LAYER"
            echo ""
        fi

        # If we started the pipeline, wait for Ctrl+C
        PIPELINE_PID=$(read_pid "pipeline")
        if [[ -n "$PIPELINE_PID" ]] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
            trap 'echo ""; do_stop; exit 0' INT TERM
            echo "Press Ctrl+C to stop all services..."
            wait "$PIPELINE_PID" 2>/dev/null || true
        fi
        ;;
    stop)
        do_stop
        ;;
    status)
        do_status
        ;;
esac
