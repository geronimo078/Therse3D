// Compute-Sort & Quad-Rendering für 3D Gaussian Splatting

struct SplatVertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> SplatVertexOutput {
  var out: SplatVertexOutput;
  let x = f32((vertex_id & 1u) << 1u) - 1.0;
  let y = f32((vertex_id & 2u)) - 1.0;
  out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
  out.uv = vec2<f32>(x, y);
  return out;
}

@fragment
fn fs_main(in: SplatVertexOutput) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) {
    discard;
  }
  let alpha = exp(-4.0 * r2);
  return vec4<f32>(1.0, 1.0, 1.0, alpha);
}

