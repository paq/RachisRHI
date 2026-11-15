{
  description = "RachisRHI development shells";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      commonPackages = pkgs: [
        pkgs.cmake
        pkgs.ninja
        pkgs.clang-tools
        pkgs.shader-slang
        pkgs.directx-shader-compiler
        pkgs.glslang
        pkgs.spirv-tools
        pkgs.vulkan-loader
        pkgs.vulkan-validation-layers
        pkgs.vulkan-tools
      ];
      darwinPackages = pkgs: [ pkgs.moltenvk ];
      linuxPackages = pkgs: [
        pkgs.gcc
        pkgs.pkg-config
        pkgs.libffi
        pkgs.wayland
        pkgs.wayland-scanner
        pkgs.wayland-protocols
        pkgs.libxau
        pkgs.libxdmcp
        pkgs.libxcb
      ];
      darwinShellHook = pkgs: ''
        unset DEVELOPER_DIR SDKROOT MACOSX_DEPLOYMENT_TARGET NIX_CFLAGS_COMPILE NIX_LDFLAGS
        export PATH="/usr/bin:$PATH"
        export CC=/usr/bin/clang
        export CXX=/usr/bin/clang++
        export OBJC=/usr/bin/clang
        export OBJCXX=/usr/bin/clang++
        export SLANGC=${pkgs.shader-slang}/bin/slangc
        export VULKAN_SDK=${pkgs.vulkan-loader}
        export VK_DRIVER_FILES=${pkgs.moltenvk}/share/vulkan/icd.d/MoltenVK_icd.json
        export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
      '';
      linuxShellHook = pkgs: ''
        export SLANGC=${pkgs.shader-slang}/bin/slangc
        export LD_LIBRARY_PATH=${
          pkgs.lib.makeLibraryPath [
            pkgs.vulkan-loader
            pkgs.wayland
            pkgs.libxcb
          ]
        }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
      '';
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages =
            commonPackages pkgs
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin (darwinPackages pkgs)
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux (linuxPackages pkgs);
          shellHook =
            pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin (darwinShellHook pkgs)
            + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux (linuxShellHook pkgs);
        };
      });
    };
}
