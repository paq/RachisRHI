#!/usr/bin/env bash
# Installs multi-user Nix on a GitHub-hosted Linux or macOS runner and puts it on PATH
# for later steps. The pinned install script verifies the SHA-256 of the Nix tarball it
# downloads, so pinning the script pins the whole installation.
#
# GITHUB_TOKEN, when set, is given to Nix for github.com so that fetching flake inputs
# is not rate-limited.
set -euo pipefail

version=2.35.6
install_sha256=6f7bf206efb54cd4941fb274930dbc74fdfc16fcb06757ef261c8d7d69cfd60e

if command -v nix >/dev/null 2>&1; then
  echo "install-nix.sh: Nix is already installed at $(command -v nix)" >&2
  exit 1
fi

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

curl --fail --silent --show-error --location --retry 5 --retry-all-errors \
  --output "$workdir/install" "https://releases.nixos.org/nix/nix-$version/install"
actual=$(shasum -a 256 "$workdir/install" | cut -d ' ' -f 1)
if [ "$actual" != "$install_sha256" ]; then
  echo "install-nix.sh: the Nix $version install script has SHA-256 $actual, expected $install_sha256" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin) cert_file=/etc/ssl/cert.pem ;;
  *) cert_file=/etc/ssl/certs/ca-certificates.crt ;;
esac

{
  echo "experimental-features = nix-command flakes"
  echo "max-jobs = auto"
  echo "ssl-cert-file = $cert_file"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "access-tokens = github.com=$GITHUB_TOKEN"
  fi
} >"$workdir/nix.conf"

sh "$workdir/install" --daemon --no-channel-add --nix-extra-conf-file "$workdir/nix.conf"

echo /nix/var/nix/profiles/default/bin >>"$GITHUB_PATH"
echo "NIX_SSL_CERT_FILE=$cert_file" >>"$GITHUB_ENV"
if [ -z "${TMPDIR:-}" ]; then
  echo "TMPDIR=$RUNNER_TEMP" >>"$GITHUB_ENV"
fi

/nix/var/nix/profiles/default/bin/nix --version
