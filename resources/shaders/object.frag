#version 330 core

layout(location = 0) out float depth_out;
layout(location = 1) out vec3 normal_out;

in VertexData {
    vec3 normal;
    float depth;
} interpolated;

void  main() {
  depth_out = interpolated.depth;
  normal_out = interpolated.normal;
}