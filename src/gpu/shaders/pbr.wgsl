// Standard PBR-Shader + 3-Punkt-Studio-Beleuchtung + Dynamic Lights + Soft Contact Shadows + ACES Tone-Mapping

struct DynamicLight {
  posOrDir: vec4<f32>,      // xyz = world pos (point) or dir (directional), w = kind (0 = point, 1 = directional)
  colorIntensity: vec4<f32>, // rgb = color, w = intensity
};

struct SceneUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,
  skyColor: vec4<f32>,
  groundColor: vec4<f32>,
  keyLightDir: vec4<f32>,
  keyLightColor: vec4<f32>,
  counts: vec4<f32>, // x = dynamicLightCount (0..8), y = shadowCasterCount (0..16)
  lights: array<DynamicLight, 8>,
  shadowCasters: array<vec4<f32>, 16>, // x = cx, y = minY, z = cz, w = radiusXZ
  lightViewProj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> modelMatrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pbrParams: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> instanceIds: array<u32>;
@group(0) @binding(5) var shadowMap: texture_depth_2d;
@group(0) @binding(6) var shadowSampler: sampler_comparison;
@group(0) @binding(7) var albedoTextures: texture_2d_array<f32>;
@group(0) @binding(8) var albedoSampler: sampler;

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

@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
  let entityId = instanceIds[in.instanceIdx];
  let model = modelMatrices[entityId];
  let worldPos = model * vec4<f32>(in.position, 1.0);
  return scene.lightViewProj * worldPos;
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

fn computeContactShadow(worldPos: vec3<f32>, N: vec3<f32>) -> f32 {
  let casterCount = i32(scene.counts.y);
  if (casterCount <= 0 || N.y < 0.5) {
    return 1.0;
  }
  var shadow: f32 = 1.0;
  for (var i = 0; i < 16; i++) {
    if (i >= casterCount) {
      break;
    }
    let c = scene.shadowCasters[i];
    let dy = c.y - worldPos.y;
    if (dy >= -0.02 && dy <= 0.35) {
      let distXZ = length(worldPos.xz - c.xz);
      let radius = max(c.w, 0.05);
      let radial = 1.0 - smoothstep(radius * 0.45, radius * 1.35, distXZ);
      let heightFade = 1.0 - smoothstep(0.0, 0.35, max(dy, 0.0));
      shadow = min(shadow, 1.0 - radial * heightFade * 0.55);
    }
  }
  return shadow;
}

fn computeDirectionalShadow(worldPos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
  let lightClip = scene.lightViewProj * vec4<f32>(worldPos, 1.0);
  let ndc = lightClip.xyz / max(lightClip.w, 1e-5);
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let inBounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0;

  let bias = max(0.0008 * (1.0 - max(dot(N, L), 0.0)), 0.00025);
  let texelSize = 1.0 / 2048.0;
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2<f32>(f32(x), f32(y)) * texelSize;
      sum += textureSampleCompareLevel(shadowMap, shadowSampler, uv + offset, ndc.z - bias);
    }
  }
  let pcf = sum / 9.0;
  return select(1.0, pcf, inBounds);
}

fn acesToneMap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hash21(p: vec2<f32>) -> f32 {
  let q = fract(p * vec2<f32>(123.34, 456.21));
  let d = dot(q, q + 45.32);
  return fract((q.x + q.y) * d);
}

fn applyProceduralMaterial(
  materialId: i32,
  worldPos: vec3<f32>,
  uv: vec2<f32>,
  baseAlbedo: vec3<f32>,
  baseRoughness: f32,
) -> vec4<f32> {
  // Compute derivatives in uniform control flow before branching on materialId
  let worldXZDeriv = max(fwidth(worldPos.xz), vec2<f32>(0.001));
  var albedo = baseAlbedo;
  var rough = baseRoughness;

  if (materialId == 1) {
    // oak-wood: procedural grain rings along world X/Z
    let p = worldPos.xz * 4.5 + vec2<f32>(worldPos.y * 1.5, 0.0);
    let warp = sin(p.x * 2.0 + sin(p.y * 3.0) * 0.6) * 0.5;
    let ring = 0.5 + 0.5 * sin((p.x * 8.0 + warp * 4.0) + p.y * 1.5);
    albedo = mix(baseAlbedo * 0.76, baseAlbedo * 1.12, ring);
    rough = clamp(baseRoughness + (ring - 0.5) * 0.12, 0.08, 1.0);
  } else if (materialId == 2) {
    // brushed-aluminum: fine directional streaks
    let streak = sin(uv.y * 260.0 + sin(uv.x * 25.0) * 1.5) * 0.5 + 0.5;
    albedo = baseAlbedo * mix(0.93, 1.05, streak);
    rough = clamp(baseRoughness + (streak - 0.5) * 0.08, 0.08, 1.0);
  } else if (materialId == 8) {
    // studio-grid: world-space 1.0m major + 0.25m minor anti-aliased grid
    let majorCoord = abs(fract(worldPos.xz - 0.5) - 0.5) / worldXZDeriv;
    let majorLine = 1.0 - min(min(majorCoord.x, majorCoord.y), 1.0);
    let minorCoord = abs(fract(worldPos.xz * 4.0 - 0.5) - 0.5) / (worldXZDeriv * 4.0);
    let minorLine = 1.0 - min(min(minorCoord.x, minorCoord.y), 1.0);
    let axisX = 1.0 - smoothstep(0.0, 0.03, abs(worldPos.z));
    let axisZ = 1.0 - smoothstep(0.0, 0.03, abs(worldPos.x));
    albedo = mix(baseAlbedo, baseAlbedo * 1.45, minorLine * 0.35);
    albedo = mix(albedo, vec3<f32>(0.36, 0.45, 0.58), majorLine * 0.75);
    albedo = mix(albedo, vec3<f32>(0.85, 0.25, 0.25), axisX * 0.65);
    albedo = mix(albedo, vec3<f32>(0.25, 0.55, 0.90), axisZ * 0.65);
  } else if (materialId == 9) {
    // carbon-fiber: diagonal twill weave
    let cell = floor(uv * 64.0);
    let diag = i32(cell.x + cell.y) % 4;
    let weave = select(0.75, 1.25, diag < 2);
    albedo = baseAlbedo * weave;
    rough = clamp(baseRoughness + select(0.08, -0.08, diag < 2), 0.12, 1.0);
  } else if (materialId == 6) {
    // matte-ceramic: subtle micro-speckle
    let n = hash21(floor(worldPos.xz * 120.0));
    albedo = baseAlbedo * mix(0.96, 1.02, n);
  }

  return vec4<f32>(albedo, rough);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let baseColor = colors[in.entityId];
  let params = pbrParams[in.entityId]; // [metalness, roughness, emissive, materialId | (textureId << 8)]
  let packedMatTex = u32(max(params.w + 0.5, 0.0));
  let materialId = i32(packedMatTex & 255u);
  let textureId = i32(packedMatTex >> 8u);
  let texSample = textureSample(albedoTextures, albedoSampler, in.uv, textureId);
  let procMat = applyProceduralMaterial(
    materialId,
    in.worldPos,
    in.uv,
    baseColor.rgb * texSample.rgb,
    clamp(params.y, 0.04, 1.0),
  );
  let albedo = procMat.rgb;
  let metalness = clamp(params.x, 0.0, 1.0);
  let roughness = procMat.a;
  let emissive = max(params.z, 0.0);

  let N = normalize(in.worldNormal);
  let V = normalize(scene.cameraPos.xyz - in.worldPos);
  let F0 = mix(vec3<f32>(0.04), albedo, metalness);

  // 3-Punkt-Studio-Beleuchtung (Key, Fill, Rim)
  let keyDir = normalize(scene.keyLightDir.xyz);
  let fillDir = normalize(vec3<f32>(-0.7, 0.4, 0.5));
  let rimDir = normalize(vec3<f32>(0.2, 0.6, -0.8));

  var Lo = vec3<f32>(0.0);
  let dirShadow = computeDirectionalShadow(in.worldPos, N, keyDir);
  Lo += evalLight(N, V, keyDir, scene.keyLightColor.rgb * 2.4, albedo, metalness, roughness, F0) * mix(0.25, 1.0, dirShadow);
  Lo += evalLight(N, V, fillDir, vec3<f32>(0.45, 0.52, 0.65), albedo, metalness, roughness, F0);
  Lo += evalLight(N, V, rimDir, vec3<f32>(0.75, 0.80, 0.90), albedo, metalness, roughness, F0);

  // Dynamic Light Entities (up to 8)
  let lightCount = i32(scene.counts.x);
  for (var i = 0; i < 8; i++) {
    if (i >= lightCount) {
      break;
    }
    let dl = scene.lights[i];
    let isDir = dl.posOrDir.w > 0.5;
    if (isDir) {
      let L = normalize(dl.posOrDir.xyz);
      Lo += evalLight(N, V, L, dl.colorIntensity.rgb * dl.colorIntensity.w, albedo, metalness, roughness, F0);
    } else {
      let toLight = dl.posOrDir.xyz - in.worldPos;
      let dist = length(toLight);
      let L = toLight / max(dist, 0.001);
      let atten = 1.0 / (1.0 + 0.14 * dist + 0.07 * dist * dist);
      Lo += evalLight(N, V, L, dl.colorIntensity.rgb * dl.colorIntensity.w * atten, albedo, metalness, roughness, F0);
    }
  }

  let contactShadow = computeContactShadow(in.worldPos, N);

  // Hemisphere Ambient
  let hemiFactor = N.y * 0.5 + 0.5;
  let ambientLight = mix(scene.groundColor.rgb, scene.skyColor.rgb, hemiFactor);
  let ambient = ambientLight * albedo * (1.0 - metalness * 0.6) * 0.35 * contactShadow;

  let hdr = ambient + Lo * mix(0.45, 1.0, contactShadow) + albedo * emissive;
  let mapped = acesToneMap(hdr);
  let srgb = pow(mapped, vec3<f32>(1.0 / 2.2));
  return vec4<f32>(srgb, baseColor.a * texSample.a);
}

@fragment
fn fs_unlit(in: VertexOutput) -> @location(0) vec4<f32> {
  return colors[in.entityId];
}
