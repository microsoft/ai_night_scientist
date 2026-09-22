#!/usr/bin/env bash
#
# Builds a runnable Day & Night Science tree:
#   1. fetches verl at the exact pinned commit
#   2. applies our patches to the 11 verl files we modified
#   3. copies our new files (overlay/) on top
#   4. grafts in the search tool from the companion repository
#
# Usage:  ./setup.sh [--dest DIR] [--verl-url URL] [--force]
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_COMMIT="$(tr -d '[:space:]' < "$HERE/VERL_COMMIT")"
COMPANION_URL="$(tr -d '[:space:]' < "$HERE/COMPANION_REPO")"
COMPANION_COMMIT="$(tr -d '[:space:]' < "$HERE/COMPANION_COMMIT")"
DEST="$HERE/build/night_ai_scientist"
VERL_URL="https://github.com/volcengine/verl.git"
FORCE=0
COMPANION=1

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)     DEST="$2"; shift 2 ;;
    --verl-url) VERL_URL="$2"; shift 2 ;;
    --force)    FORCE=1; shift ;;
    --no-companion) COMPANION=0; shift ;;
    --companion-dir) COMPANION_DIR="$2"; shift 2 ;;
    -h|--help)  sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"

# ---------------------------------------------------------------- 1. fetch verl
if [ -e "$DEST" ]; then
  [ "$FORCE" = 1 ] || die "$DEST already exists (pass --force to replace it)"
  say "removing existing $DEST"
  rm -rf "$DEST"
fi

say "fetching verl @ ${VERL_COMMIT:0:12} into $DEST"
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin "$VERL_URL"
# --depth 1 on an explicit SHA: supported by GitHub, keeps the download small
if ! git -C "$DEST" fetch -q --depth 1 origin "$VERL_COMMIT"; then
  warn "shallow fetch of the pinned SHA failed; falling back to a full fetch"
  git -C "$DEST" fetch -q origin
fi
git -C "$DEST" checkout -q "$VERL_COMMIT"
say "verl checked out at $(git -C "$DEST" rev-parse --short HEAD)"

# --------------------------------------------------- 2. patch the 11 verl files
say "applying patches to verl files we modified"
# VERL_COMMIT pins the exact tree these patches were generated against, so a
# failure here means the commit or the remote was changed, not a stale patch.
applied=0
for p in "$HERE"/patches/*.patch; do
  [ -e "$p" ] || continue
  name="$(basename "$p")"
  git -C "$DEST" apply --3way --whitespace=nowarn "$p" 2>/dev/null \
    || die "$name failed to apply against ${VERL_COMMIT:0:12}"
  printf '    ok        %s\n' "$name"
  applied=$((applied+1))
done
say "patches: $applied applied"

# ------------------------------------------------------- 3. copy our new files
say "copying our new files from overlay/"
# Overlay never overwrites a patched file: every overlay path is new relative
# to upstream verl.
# -a preserves our modes (the .sh launchers are already executable in overlay/).
# Deliberately no blanket chmod here: that would rewrite the mode of every
# pristine verl script and bury our 11 real edits in `git status` noise.
cp -a "$HERE/overlay/." "$DEST/"

# ------------------------------------------- 4. fetch the companion services
# The arXiv retrieval service and synthetic proposal generation live in their own
# repository; they are standalone and talk to training over HTTP only. That repo
# is private, so this needs either an SSH key or a credential helper with access.
# No access? Point COMPANION_DIR at a local clone, or pass --no-companion.
if [ "$COMPANION" = 1 ]; then
  rm -rf "$DEST/companion"
  fetched=0
  if [ -n "${COMPANION_DIR:-}" ]; then
    say "using companion services from $COMPANION_DIR"
    if [ -d "$COMPANION_DIR" ]; then
      cp -a "$COMPANION_DIR" "$DEST/companion"
      rm -rf "$DEST/companion/.git"
      fetched=1
    else
      warn "COMPANION_DIR=$COMPANION_DIR does not exist"
    fi
  else
    say "fetching companion services @ ${COMPANION_COMMIT:0:12}"
    # SSH first: the repo is private, so anonymous HTTPS cannot work.
    # GIT_TERMINAL_PROMPT=0 makes HTTPS fail fast instead of blocking on a
    # username prompt when no credential helper is configured.
    COMPANION_SSH="$(printf '%s' "$COMPANION_URL" | sed -E 's#^https://github.com/#git@github.com:#')"
    for url in "$COMPANION_SSH" "$COMPANION_URL"; do
      rm -rf "$DEST/companion"
      if GIT_TERMINAL_PROMPT=0 git clone -q "$url" "$DEST/companion" 2>/dev/null \
         && git -C "$DEST/companion" checkout -q "$COMPANION_COMMIT" 2>/dev/null; then
        fetched=1
        break
      fi
    done
  fi
  if [ "$fetched" = 1 ]; then
    say "companion services at $DEST/companion"
    # verl_tool/ mirrors destination paths, so this lands the search tool at
    # verl/tools/arxiv_search_tool.py and verl/tools/utils/arxiv_utils.py.
    if [ -d "$DEST/companion/verl_tool" ]; then
      cp -a "$DEST/companion/verl_tool/." "$DEST/"
      for f in verl/tools/arxiv_search_tool.py verl/tools/utils/arxiv_utils.py; do
        [ -f "$DEST/$f" ] || die "companion did not provide $f"
      done
      say "search tool grafted into the verl tree"
    else
      die "companion has no verl_tool/ - is COMPANION_COMMIT older than that change?"
    fi
  else
    rm -rf "$DEST/companion"
    printf '\033[1;31m[fail]\033[0m %s\n' "could not obtain the companion repository at ${COMPANION_COMMIT:0:12}" >&2
    printf '        %s\n' \
      "$COMPANION_URL is private - you need an SSH key or a credential" \
      "helper with access to it." \
      "" \
      "It is not optional: it provides verl/tools/arxiv_search_tool.py and" \
      "verl/tools/utils/arxiv_utils.py, which the reward manager imports. A tree" \
      "without them will not start." \
      "" \
      "Already have a clone?" \
      "    COMPANION_DIR=/path/to/synthetic_proposals $0 --dest \"$DEST\" --force" >&2
    exit 1
  fi
fi

# ------------------------------------------------------------------- 5. verify
say "verifying"
missing=0
while IFS= read -r f; do
  [ -f "$DEST/$f" ] || { warn "missing: $f"; missing=$((missing+1)); }
done < <(cd "$HERE/overlay" && find . -type f | sed 's|^\./||')
[ "$missing" = 0 ] || die "$missing overlay file(s) did not land"

n_overlay=$(cd "$HERE/overlay" && find . -type f | wc -l)
say "$n_overlay overlay files in place; tree is at $DEST"

if [ -d "$DEST/companion" ]; then
  COMPANION_STEP="  # start the retrieval service and leave it running (separate shell):
  cd companion && pip install -r requirements.txt && bash retrieval/arxiv_emb_retrieval.sh

"
else
  COMPANION_STEP="  # INCOMPLETE TREE: the companion repo was skipped, so verl/tools/arxiv_search_tool.py
  # and verl/tools/utils/arxiv_utils.py are missing and the reward manager will fail to
  # import. Re-run without --no-companion, or with COMPANION_DIR=/path/to/synthetic_proposals.

"
fi

cat <<EOF

Next steps
  cd $DEST
  pip install -e .                      # install verl + our modules
  pip install -r requirements-cuda.txt  # or requirements-npu.txt

Then, from that directory:
  python examples/data_preprocess/proposal_gen.py     # build the dataset
${COMPANION_STEP}  bash examples/sglang_multiturn/day_night/run_qwen2.5-3b_instruct_proposal_gen_multiturn.sh

See README.md for what lives where.
EOF
