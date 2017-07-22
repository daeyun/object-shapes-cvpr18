with import <nixpkgs> {};

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
    (sqlite // { meta = ( sqlite.meta // { outputsToInstall = ["out" "dev"]; } ); })
    (boost // { meta = ( boost.meta // { outputsToInstall = ["out" "dev"]; } ); })
  ];
}
