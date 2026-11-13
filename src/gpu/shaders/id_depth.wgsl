// Multi-Pass MRT Shader for Object-ID Picking (r32uint) & Linearized Depth Maps (r32float)

struct CaptureUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  nearFar: vec4<f32>, // x = near, y = far
};

@group(0) @binding(0) var<uniform> capture: CaptureUniforms;
@group(0) @binding(1) var<storage, read> modelMatrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> instanceIds: array<u32>;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @builtin(instance_index) instanceIdx: u32,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) @interpolate(flat) entityId: u32,
};

struct FragmentOutput {
  @location(0) objectId: u32,
  @location(1) linearDepth: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  let entityId = instanceIds[in.instanceIdx];
  let model = modelMatrices[entityId];
  let worldPos = model * vec4<f32>(in.position, 1.0);

  var out: VertexOutput;
  out.clipPos = capture.viewProj * worldPos;
  out.worldPos = worldPos.xyz;
  out.entityId = entityId + 1u; // 0 is reserved for background
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  let dist = length(in.worldPos - capture.cameraPos.xyz);
  let nearPlane = capture.nearFar.x;
  let farPlane = max(capture.nearFar.y, nearPlane + 0.001);
  let linear = clamp((dist - nearPlane) / (farPlane - nearPlane), 0.0, 1.0);

  var out: FragmentOutput;
  out.objectId = in.entityId;
  out.linearDepth = linear;
  return out;
}
