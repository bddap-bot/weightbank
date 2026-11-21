let
  pkgs = import ./nixpkgs.nix { config.allowUnfree = true; };
  python = pkgs.python3.withPackages (ps: [
    ps.torch-bin
    ps.numpy
    ps.matplotlib
    ps.mypy
    ps.pytest
    ps.safetensors
    ps.tokenizers
    ps.pyarrow
    ps.scipy
  ]);
in
pkgs.mkShell { packages = [ python ]; }
