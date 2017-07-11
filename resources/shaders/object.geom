#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

uniform mat4 projection;
uniform float scale;

out VertexData {
    vec3 normal;
    float depth;
} interpolated;

void main() {
  // Computes the face normal in camera coordinates.
  vec3 ab = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
  vec3 ac = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;

  // Counter-clockwise normal.
  interpolated.normal = normalize(cross(ac, ab));

  // Flip towards camera, assuming viewing direction is (0, 0, -1).
  if (interpolated.normal[2] < -1e-8) {
    interpolated.normal = normalize(-interpolated.normal);
  }

  // Apply projection matrix and send to fragment shader.
  for(int i = 0; i < gl_in.length(); i++) {
    interpolated.depth = scale * abs(gl_in[i].gl_Position[2]);
    gl_Position = projection * gl_in[i].gl_Position;
    gl_Position[0] *= scale;
    gl_Position[1] *= scale;
    EmitVertex();
  }
}