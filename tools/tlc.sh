#!/usr/bin/env bash
# Model-checks the TLA+ specs in formal/tla with TLC (formal/tla/README.md).
#
#   bash tools/tlc.sh                  every configuration in formal/tla
#   bash tools/tlc.sh --small          those not marked `\* Size: large`
#   bash tools/tlc.sh <file.cfg>...    only these configurations
#
# A configuration formal/tla/<Module>.cfg or formal/tla/<Module>-<label>.cfg is
# checked against formal/tla/<Module>.tla. Its header comment states its bounds
# and the verdict TLC must reach, in one of these forms:
#
#   \* Expect: pass                  no error
#   \* Expect: invariant <Name>      the invariant <Name> is violated
#   \* Expect: property <Name>       the property <Name> is violated
#   \* Expect: deadlock              TLC reaches a deadlock
#
# A violation counts in an initial state as in any other. TLC checks a property
# of the form []P like an invariant and reports it as one, so `property` accepts
# that report too, as well as those for temporal and action properties.
#
# A configuration that expects a violation keeps a counterexample checked: the
# model with one rule of the design turned off. The script fails unless every
# configuration reaches its expected verdict, and prints each one's state
# counts and time. A header line `\* Size: large` marks a configuration that
# takes more than a minute on a desktop machine; --small leaves those out.
#
# TLA2TOOLS must name tla2tools.jar (1.8.0 or later); Java comes from JAVA,
# JAVA_HOME, or `java` on PATH. Each run is limited to TLC_TIMEOUT seconds (1800 by default) with stdin
# closed, and uses TLC_WORKERS workers (`auto` by default). TLC's state files
# go to a temporary directory that is removed on exit.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
spec_dir="$repo_root/formal/tla"
jar="${TLA2TOOLS:-}"
java="${JAVA:-${JAVA_HOME:+$JAVA_HOME/bin/java}}"
java="${java:-java}"
limit="${TLC_TIMEOUT:-1800}"
workers="${TLC_WORKERS:-auto}"

if [ -z "$jar" ]; then
  echo "tlc.sh: set TLA2TOOLS to the path of tla2tools.jar (1.8.0 or later)" >&2
  exit 1
fi
if [ ! -f "$jar" ]; then
  echo "tlc.sh: no tla2tools.jar at $jar; set TLA2TOOLS to the path of tla2tools.jar" >&2
  exit 1
fi
if ! command -v "$java" >/dev/null 2>&1 || ! "$java" -version >/dev/null 2>&1; then
  echo "tlc.sh: no working Java at $java; set JAVA or JAVA_HOME to a Java 11 or later binary" >&2
  exit 1
fi

# Java on Windows needs native paths; Git Bash has cygpath, POSIX hosts do not.
native() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s\n' "$1"
  fi
}

# macOS has no `timeout` on default PATH; fall back to a process-group alarm in Perl.
run_bounded() {
  local secs=$1
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout -k 5 "$secs" "$@"
  else
    perl -e '
      my $secs = shift @ARGV;
      my $pid = fork();
      die "fork failed: $!" unless defined $pid;
      if ($pid == 0) {
        setpgrp(0, 0);
        exec @ARGV or exit 127;
      }
      setpgrp($pid, $pid);
      local $SIG{ALRM} = sub {
        kill "TERM", -$pid;
        select(undef, undef, undef, 0.2);
        kill "KILL", -$pid;
        waitpid($pid, 0);
        exit 124;
      };
      alarm($secs);
      waitpid($pid, 0);
      my $st = $?;
      alarm(0);
      if ($st & 127) { exit(128 + ($st & 127)); }
      exit($st >> 8);
    ' "$secs" "$@"
  fi
}

tmpdir="${TMPDIR:-/tmp}"
work=$(mktemp -d "${tmpdir%/}/rachis-tlc.XXXXXX")
trap 'rm -rf "$work"' EXIT
jar_native=$(native "$(cd "$(dirname "$jar")" && pwd)/$(basename "$jar")")

small=0
if [ "${1:-}" = --small ]; then
  small=1
  shift
  if [ "$#" -ne 0 ]; then
    echo "tlc.sh: --small takes no configurations" >&2
    exit 2
  fi
fi

configs=()
if [ "$#" -eq 0 ]; then
  for cfg in "$spec_dir"/*.cfg; do
    [ -e "$cfg" ] || continue
    if [ "$small" -eq 1 ] && grep -q '^\\\* Size: large' "$cfg"; then
      echo "${cfg#"$repo_root"/}: skipped, large"
      continue
    fi
    configs+=("$cfg")
  done
  if [ "${#configs[@]}" -eq 0 ]; then
    echo "tlc.sh: no configurations in $spec_dir" >&2
    exit 1
  fi
else
  for cfg in "$@"; do
    case "$cfg" in
      /*) ;;
      *) cfg="$PWD/$cfg" ;;
    esac
    configs+=("$cfg")
  done
fi

failed=0
for cfg in "${configs[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "tlc.sh: no configuration at $cfg" >&2
    failed=1
    continue
  fi
  name=$(basename "$cfg" .cfg)
  module=${name%%-*}
  dir=$(cd "$(dirname "$cfg")" && pwd)
  shown="${dir#"$repo_root"/}/$name.cfg"
  if [ ! -f "$dir/$module.tla" ]; then
    echo "$shown: no $module.tla next to it" >&2
    failed=1
    continue
  fi
  expect=$(sed -n 's/^\\\* Expect: *//p' "$cfg" | head -n 1 | tr -d '\r')
  # Each verdict is one of TLC's exit statuses with the line it prints for it.
  case "$expect" in
    pass) wants=("0|Model checking completed. No error has been found.") ;;
    "invariant "*)
      want=${expect#invariant }
      wants=("12|Error: Invariant $want is violated." "12|Error: Invariant $want is violated by the initial state:")
      ;;
    "property "*)
      want=${expect#property }
      wants=(
        "13|Error: Temporal property $want was violated."
        "13|Error: Action property $want is violated."
        "12|Error: Action property $want is violated."
        "12|Error: Invariant $want is violated."
        "12|Error: Invariant $want is violated by the initial state:"
      )
      ;;
    deadlock) wants=("11|Error: Deadlock reached.") ;;
    *)
      echo "$shown: no '\\* Expect:' line, or an unknown verdict '$expect'" >&2
      failed=1
      continue
      ;;
  esac
  log="$work/$name.log"
  status=0
  (cd "$dir" && run_bounded "$limit" "$java" -XX:+UseParallelGC -cp "$jar_native" tlc2.TLC \
    -workers "$workers" -noGenerateSpecTE -metadir "$(native "$work/states-$name")" \
    -config "$name.cfg" "$module.tla" </dev/null >"$log.raw" 2>&1) || status=$?
  # Java on Windows ends its lines with CRLF.
  tr -d '\r' <"$log.raw" >"$log"
  summary=$(sed -n 's/^\([0-9,]*\) states generated, \([0-9,]*\) distinct states found.*/\2 distinct states (\1 generated)/p' "$log" | tail -n 1)
  depth=$(sed -n 's/^The depth of the complete state graph search is \([0-9]*\)\.$/\1/p' "$log" | tail -n 1)
  took=$(sed -n 's/^Finished in \(.*\) at .*/\1/p' "$log" | tail -n 1)
  verdict=0
  for w in "${wants[@]}"; do
    if [ "$status" -eq "${w%%|*}" ] && grep -qxF "${w#*|}" "$log"; then
      verdict=1
    fi
  done
  if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    echo "$shown: FAIL, TLC did not finish within ${limit}s (TLC_TIMEOUT)"
    tail -n 5 "$log"
    failed=1
  elif [ "$verdict" -eq 1 ]; then
    echo "$shown: $expect as expected; ${summary:-no state count}, depth ${depth:-?}, ${took:-?}"
  else
    echo "$shown: FAIL, expected '$expect' but TLC exited with $status"
    cat "$log"
    failed=1
  fi
done
exit "$failed"
