with import <nixpkgs> {config.allowUnfree = true;};

let
 mesa_noglu = (mesa_noglu.override {
    enableTextureFloats = true;
  }).drivers;
in
stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs;};
  buildInputs = [
    opencv3
    c-blosc
    eigen
    assimp
    google-gflags
    gmock
    glog
    protobuf3_2
    gcc6
    snappy
    zlib
    (glibc // { meta = ( glibc.meta // { outputsToInstall = [ "out" "dev" "static" "debug" "bin" ]; } ); })
    (libdrm // { meta = ( libdrm.meta // { outputsToInstall = [ "out" "dev" ]; } ); })
    (xorg.libxcb // { meta = ( xorg.libxcb.meta // { outputsToInstall = [ "out" "dev" ]; } ); })
    (mesa_glu // { meta = ( mesa_glu.meta // { outputsToInstall = [ "out" "dev" ]; } ); })
    (mesa_noglu // { meta = ( mesa_noglu.meta // { outputsToInstall = [ "out" "dev" "drivers" "osmesa" ]; } ); })
    (xorg.libX11 // { meta = ( xorg.libX11.meta // { outputsToInstall = [ "out" "dev" ]; } ); })
    (xorg.xproto // { meta = ( xorg.xproto.meta // { outputsToInstall = [ "out" "doc" ]; } ); })
    (sqlite // { meta = ( sqlite.meta // { outputsToInstall = ["out" "dev"]; } ); })
    (boost // { meta = ( boost.meta // { outputsToInstall = ["out" "dev"]; } ); })
  ];
}