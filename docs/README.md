# RachisRHI Documentation

RachisRHI is a C++23 GPU abstraction with a versioned C11 ABI, targeting Direct3D 12 and Vulkan on Windows, Vulkan on Linux, and Metal 4 on Apple silicon macOS. Public headers contain no Win32, DXGI, Direct3D 12, Vulkan, or Metal types.

The current package version is `0.3.1`. Until `1.0.0`, the C++ API may change between minor releases; the C ABI (`RACHIS_ABI_VERSION`, currently `3`) changes only when its version number is bumped and verified before device creation.

## Core guides

| Guide | Read this for |
| --- | --- |
| [Platforms](platforms.md) | Supported hosts, optional feature matrix (`DeviceCapabilities`), Vulkan profile, validation layers, and backend-specific quirks |
| [Building](building.md) | Required tools, Nix shell, CMake presets and options, running test tiers, packaging/installing, and embedding the source tree |
| [Using the library](using.md) | CMake targets, Windows DLLs, device creation, shaders (`rachis_add_shader`), vertex layouts, surfaces/swapchains, lifetimes, barriers, binding pools, threading, queries, and device loss |
| [C ABI](c-abi.md) | Using `Rachis/rachis.h` (`Rachis::C`) across module boundaries: ABI versioning, descriptor checks, coverage, texture leases, timelines, and threading |

## Repository guides and references

These documents live in the source repository (installed packages include `README.md`, `CHANGELOG.md`, `docs/`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`):

| Document | Contents |
| --- | --- |
| [Examples](../examples/README.md) | Standalone compute and offscreen-triangle examples running across every backend |
| [Tests](../tests/README.md) | Test directory layout, `size:` and `requires:` tags, package tests, and test-authoring rules |
| [TLA+ models](../formal/tla/README.md) | 97 formal safety models checked with TLC, counterexamples, and trusted boundaries |
| [Changelog](../CHANGELOG.md) | Release history |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Bundled dependency versions and license locations |
