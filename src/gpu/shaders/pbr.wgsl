// Standard PBR-Shader + 3-Punkt-Studio-Beleuchtung + Auto-Instancing + ACES Tone-Mapping

struct SceneUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  skyColor: vec4<f32>,
  groundColor: vec4<f32>,
  keyLightDir: vec4<f32>,
  keyLightColor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> modelMatrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pbrParams: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> instanceIds: array<u32>;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @builtin(instance_index) instanceIdx: u32,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) @interpolate(flat) entityId: u32,
};

const PI: f32 = 3.14159265359;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  let entityId = instanceIds[in.instanceIdx];
  let model = modelMatrices[entityId];
  let worldPos = model * vec4<f32>(in.position, 1.0);

  // Transform normal using upper 3x3 of model matrix (approx for uniform/non-uniform scale)
  let nMat = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
  let worldNormal = normalize(nMat * in.normal);

  var out: VertexOutput;
  out.clipPos = scene.viewProj * worldPos;
  out.worldPos = worldPos.xyz;
  out.worldNormal = worldNormal;
  out.uv = in.uv;
  out.entityId = entityId;
  return out;
}

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  let denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / max(PI * denom * denom, 1e-6);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return NdotV / max(NdotV * (1.0 - k) + k, 1e-6);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
         geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn evalLight(
  N: vec3<f32>,
  V: vec3<f32>,
  L: vec3<f32>,
  radiance: vec3<f32>,
  albedo: vec3<f32>,
  metalness: f32,
  roughness: f32,
  F0: vec3<f32>,
) -> vec3<f32> {
  let NdotL = max(dot(N, L), 0.0);
  if (NdotL <= 0.0) {
    return vec3<f32>(0.0);
  }
  let H = normalize(V + L);
  let NDF = distributionGGX(N, H, roughness);
  let G = geometrySmith(N, V, L, roughness);
  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

  let kS = F;
  let kD = (vec3<f32>(1.0) - kS) * (1.0 - metalness);

  let NdotV = max(dot(N, V), 0.001);
  let specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, 0.001);
  return (kD * albedo / PI + specular) * radiance * NdotL;
}

fn acesToneMap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let baseColor = colors[in.entityId];
  let params = pbrParams[in.entityId]; // [metalness, roughness, emissive, materialId]
  let albedo = baseColor.rgb;
  let metalness = clamp(params.x, 0.0, 1.0);
  let roughness = clamp(params.y, 0.04, 1.0);
  let emissive = max(params.z, 0.0);

  let N = normalize(in.worldNormal);
  let V = normalize(scene.cameraPos.xyz - in.worldPos);
  let F0 = mix(vec3<f32>(0.04), albedo, metalness);

  // 3-Punkt-Studio-Beleuchtung (Key, Fill, Rim)
  let keyDir = normalize(scene.keyLightDir.xyz);
  let fillDir = normalize(vec3<f32>(-0.7, 0.4, 0.5));
  let rimDir = normalize(vec3<f32>(0.2, 0.6, -0.8));

  var Lo = vec3<f32>(0.0);
  Lo += evalLight(N, V, keyDir, scene.keyLightColor.rgb * 2.6, albedo, metalness, roughness, F0);
  Lo += evalLight(N, V, fillDir, vec3<f32>(0.45, 0.52, 0.65), albedo, metalness, roughness, F0);
  Lo += evalLight(N, V, rimDir, vec3<f32>(0.75, 0.80, 0.90), albedo, metalness, roughness, F0);

  // Hemisphere Ambient
  let hemiFactor = N.y * 0.5 + 0.5;
  let ambientLight = mix(scene.groundColor.rgb, scene.skyColor.rgb, hemiFactor);
  let ambient = ambientLight * albedo * (1.0 - metalness * 0.6) * 0.35;

  let hdr = ambient + Lo + albedo * emissive;
  let mapped = acesToneMap(hdr);
  let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(srgb, baseColor.a);
}
