#!/bin/bash
# ============================================================
# Ghidra headless analysis runner for ~/Downloads/c_program
# Produces reverse-engineering artifacts per binary under
#   ~/Downloads/c_program/reverse_eng/<variant>/<binary>/
# ============================================================
set -u

export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

GHIDRA_DIR="$HOME/Downloads/ghidra_12.0.4_PUBLIC"
ANALYZE="$GHIDRA_DIR/support/analyzeHeadless"
SRC_ROOT="$HOME/Downloads/c_program"
OUT_ROOT="$SRC_ROOT/reverse_eng"
PROJ_DIR="/tmp/ghidra_headless_proj"
SCRIPT_DIR="$SRC_ROOT"     # ghidra_dump.py lives here

if [ ! -x "$ANALYZE" ]; then
    echo "ERROR: analyzeHeadless not found at $ANALYZE"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/GhidraDump.java" ]; then
    echo "ERROR: post-script not found at $SCRIPT_DIR/GhidraDump.java"
    exit 1
fi

rm -rf "$OUT_ROOT" "$PROJ_DIR"
mkdir -p "$OUT_ROOT" "$PROJ_DIR"

count=0
failed=0
declare -a failures

for variant in stripped debug; do
    src_dir="$SRC_ROOT/build/$variant"
    [ -d "$src_dir" ] || { echo "skip: $src_dir"; continue; }

    for bin_path in "$src_dir"/*; do
        # Only regular executable Mach-O files
        [ -d "$bin_path" ] && continue
        base=$(basename "$bin_path")
        [ "$base" = ".DS_Store" ] && continue
        if ! file "$bin_path" | grep -q "Mach-O"; then
            echo "skip (not Mach-O): $bin_path"
            continue
        fi

        out_dir="$OUT_ROOT/$variant/$base"
        mkdir -p "$out_dir"
        log="$out_dir/ghidra_headless.log"
        proj_name="hl_${variant}_${base}"

        count=$((count + 1))
        echo ""
        echo "=================================================="
        echo " [$count] $variant / $base"
        echo "=================================================="
        echo "  -> $out_dir"

        "$ANALYZE" "$PROJ_DIR" "$proj_name" \
            -import "$bin_path" \
            -scriptPath "$SCRIPT_DIR" \
            -postScript GhidraDump.java "$out_dir" \
            -deleteProject \
            -analysisTimeoutPerFile 300 \
            > "$log" 2>&1

        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  FAILED (rc=$rc) — see $log"
            failed=$((failed + 1))
            failures+=("$variant/$base (rc=$rc)")
        else
            # Quick sanity count
            fc=$(wc -l < "$out_dir/functions.txt" 2>/dev/null || echo 0)
            dc=$(wc -l < "$out_dir/decompiled.c" 2>/dev/null || echo 0)
            sc=$(wc -l < "$out_dir/strings.txt" 2>/dev/null || echo 0)
            echo "  OK  functions=$fc  decomp_lines=$dc  strings=$sc"
        fi
    done
done

# Also copy sources for reference
if [ -d "$SRC_ROOT/src" ]; then
    mkdir -p "$OUT_ROOT/original_sources"
    cp "$SRC_ROOT/src"/*.c "$OUT_ROOT/original_sources/" 2>/dev/null || true
fi

# Top-level index
{
    echo "# Ghidra Headless Reverse Engineering Output"
    echo ""
    echo "Generated: $(date)"
    echo "Ghidra: $GHIDRA_DIR"
    echo "Binaries processed: $count"
    echo "Failures: $failed"
    echo ""
    echo "## Layout"
    echo ""
    echo '```'
    (cd "$OUT_ROOT" && find . -maxdepth 2 -type d | sort)
    echo '```'
    echo ""
    echo "## Per-binary artifacts"
    echo ""
    echo "Each binary folder contains:"
    echo "- summary.txt       program metadata, memory map"
    echo "- functions.txt     function list (entry, name, signature, size)"
    echo "- decompiled.c      Ghidra decompiler output for every function"
    echo "- disassembly.txt   instruction listing per function"
    echo "- strings.txt       defined strings"
    echo "- symbols.txt       all symbols"
    echo "- imports.txt       external/imported symbols"
    echo "- xrefs.txt         cross-references to function entries"
    echo "- ghidra_headless.log  full Ghidra headless log"
} > "$OUT_ROOT/README.md"

echo ""
echo "=================================================="
echo " Done. Processed $count binaries, $failed failed."
echo " Output: $OUT_ROOT"
echo "=================================================="
if [ ${#failures[@]} -gt 0 ]; then
    echo "Failures:"
    for f in "${failures[@]}"; do echo "  - $f"; done
fi
