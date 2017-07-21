with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs;};
  buildInputs = [
    opencv
    c-blosc
    eigen
    assimp
    google-gflags
    gmock
    glog
    protobuf3_2
    sqlite
  ];
}