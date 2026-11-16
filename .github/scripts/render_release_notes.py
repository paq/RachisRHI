#!/usr/bin/env python3
"""Validate tag/package metadata and render GitHub release notes from CHANGELOG.md."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


TAG_PATTERN = re.compile(r"^v([0-9]+\.[0-9]+\.[0-9]+)(-[0-9A-Za-z.-]+)?$")
CMAKE_VERSION_PATTERN = re.compile(
    r"set\s*\(\s*RACHIS_PACKAGE_VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)\s*\)"
)
HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$")
LINK_PATTERN = re.compile(r"(!?\[[^\]]+\]\()([^)\s]+)(\))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate release tag against CMakeLists.txt and extract CHANGELOG.md notes."
    )
    parser.add_argument("--tag", required=True, help="Release tag (e.g. v0.2.0)")
    parser.add_argument("--repo", required=True, help="GitHub repository (owner/name)")
    parser.add_argument("--output", required=True, help="Output Markdown path")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root directory (defaults to current working directory)",
    )
    return parser.parse_args()


def read_cmake_package_version(cmake_lists: pathlib.Path) -> str:
    content = cmake_lists.read_text(encoding="utf-8")
    match = CMAKE_VERSION_PATTERN.search(content)
    if match is None:
        raise ValueError(f"Could not find RACHIS_PACKAGE_VERSION in {cmake_lists}")
    return match.group(1)


def extract_changelog_section(
    changelog_text: str, candidates: list[str]
) -> tuple[str, str]:
    lines = changelog_text.splitlines()
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None

    for line in lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match is not None:
            current_heading = heading_match.group(1).strip()
            sections.setdefault(current_heading, [])
            continue
        if current_heading is not None:
            sections[current_heading].append(line)

    footer_lines: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("Until 1.0.0,"):
            footer_lines = lines[index:]
            break

    for candidate in candidates:
        if candidate in sections:
            body_lines = list(sections[candidate])
            while body_lines and not body_lines[0].strip():
                body_lines.pop(0)
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()

            if footer_lines:
                footer_joined = "\n".join(footer_lines).strip()
                body_joined = "\n".join(body_lines).strip()
                if body_joined.endswith(footer_joined):
                    body_joined = body_joined[: -len(footer_joined)].rstrip()
                    body_lines = body_joined.splitlines()

            if not any(item.strip() for item in body_lines):
                raise ValueError(f"CHANGELOG.md section '## {candidate}' is empty")

            footer_text = "\n".join(footer_lines).strip()
            return "\n".join(body_lines).strip(), footer_text

    raise ValueError(
        f"CHANGELOG.md has no section matching {', '.join(repr('## ' + c) for c in candidates)}"
    )


def rewrite_relative_links(
    text: str, repo: str, tag: str, repo_root: pathlib.Path
) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, target, suffix = match.groups()
        if prefix.startswith("!["):
            return match.group(0)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        target_path = target.split("#", 1)[0].lstrip("./")
        resolved = repo_root / target_path
        kind = "tree" if resolved.is_dir() else "blob"
        normalized_target = target.lstrip("./")
        return f"{prefix}https://github.com/{repo}/{kind}/{tag}/{normalized_target}{suffix}"

    return LINK_PATTERN.sub(replace, text)


def main() -> int:
    args = parse_args()
    tag_match = TAG_PATTERN.match(args.tag)
    if tag_match is None:
        print(
            f"error: tag '{args.tag}' must match v<major>.<minor>.<patch>[-prerelease]",
            file=sys.stderr,
        )
        return 1

    base_version = tag_match.group(1)
    prerelease_suffix = tag_match.group(2) or ""
    full_version = f"{base_version}{prerelease_suffix}"
    repo_root = pathlib.Path(args.root).resolve()

    cmake_version = read_cmake_package_version(repo_root / "CMakeLists.txt")
    if cmake_version != base_version:
        print(
            f"error: tag '{args.tag}' expects RACHIS_PACKAGE_VERSION '{base_version}', "
            f"but CMakeLists.txt sets '{cmake_version}'",
            file=sys.stderr,
        )
        return 1

    changelog_text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    candidates = [full_version] if not prerelease_suffix else [full_version, base_version]
    body_text, footer_text = extract_changelog_section(changelog_text, candidates)

    header = (
        "Source release. No prebuilt packages are attached yet; build and install from source as "
        f"[Installing](https://github.com/{args.repo}/blob/{args.tag}/docs/building.md#installing) describes."
    )
    rendered_body = rewrite_relative_links(body_text, args.repo, args.tag, repo_root)

    parts = [header, rendered_body]
    if footer_text:
        parts.append(footer_text)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
