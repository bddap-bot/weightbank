let
  pkgs = import ./nixpkgs.nix { config.allowUnfree = true; };
  python = pkgs.python3.withPackages (ps: [
    ps.torch-bin
    ps.numpy
    ps.matplotlib
    ps.mypy
    ps.pytest
  ]);
in
pkgs.mkShell { packages = [ python ]; }
