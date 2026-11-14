// Volumetric Raymarched Clouds & Exponential Height Fog Pass (Depth-Aware)

struct VolumetricUniforms {
  invViewProj: mat4x4<f32>,
  cameraPos: vec4<f32>,   // xyz = camera world pos, w = elapsed time
  sunDir: vec4<f32>,      // xyz = normalized dir toward sun, w = fogDensity
  fogColor: vec4<f32>,    // rgb = fog/scatter tint, w = fogHeight
  cloudParams: vec4<f32>, // x = cloudDensity, y = cloudCoverage, z = cloudMode (0=planar, 1=spherical), w = cloudScale
  cloudBounds: vec4<f32>, // xyz = planetCenter, w = planetRadius
  cloudLayer: vec4<f32>,  // x = minAlt, y = maxAlt, z = cloudSpeed, w = scatterG
  sunScreen: vec4<f32>,   // xy = sun screen UV, z = sunNdcZ (<0 if behind camera), w = godRaysIntensity
};

@group(0) @binding(0) var<uniform> vol: VolumetricUniforms;
@group(0) @binding(1) var sceneColorTex: texture_2d<f32>;
@group(0) @binding(2) var sceneDepthTex: texture_depth_multisampled_2d;

struct VsOut {
  @builtin(position) fragPos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_volumetric(@builtin(vertex_index) vid: u32) -> VsOut {
  // Fullscreen triangle
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  let p = pos[vid];
  var out: VsOut;
  out.fragPos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

fn hash31(p: vec3<f32>) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn noise3(x: vec3<f32>) -> f32 {
  let i = floor(x);
  let f = fract(x);
  let u = f * f * (3.0 - 2.0 * f);

  let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
  let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

  let x00 = mix(n000, n100, u.x);
  let x10 = mix(n010, n110, u.x);
  let x01 = mix(n001, n101, u.x);
  let x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
  var val = 0.0;
  var amp = 0.5;
  var freq = 1.0;
  for (var i = 0; i < 4; i++) {
    val += amp * noise3(p * freq);
    freq *= 2.07;
    amp *= 0.5;
  }
  return val;
}

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  let denom = max(1.0 + g2 - 2.0 * g * cosTheta, 0.001);
  return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

fn sampleCloudDensity(worldPos: vec3<f32>) -> f32 {
  let density = vol.cloudParams.x;
  if (density <= 0.0001) {
    return 0.0;
  }

  let coverage = clamp(vol.cloudParams.y, 0.0, 1.0);
  let mode = vol.cloudParams.z;
  let scale = max(vol.cloudParams.w, 0.05);
  let minAlt = vol.cloudLayer.x;
  let maxAlt = max(vol.cloudLayer.y, minAlt + 0.05);
  let t = vol.cameraPos.w * vol.cloudLayer.z;

  var heightFrac = 0.0;
  var samplePos = worldPos * scale;

  if (mode > 0.5) {
    // Spherical planetary cloud shell: uniform base rotation + bounded gentle shear (never winds into lines)
    let rel = worldPos - vol.cloudBounds.xyz;
    let r = length(rel);
    let rMin = vol.cloudBounds.w + minAlt;
    let rMax = vol.cloudBounds.w + maxAlt;
    if (r < rMin || r > rMax) {
      return 0.0;
    }
    heightFrac = (r - rMin) / (rMax - rMin);
    let latNorm = rel.y / max(r, 1e-4);
    let windAngle = t * 0.28 + sin(t * 0.18 + latNorm * 2.5) * 0.14;
    let c = cos(windAngle);
    let s = sin(windAngle);
    let spun = vec3<f32>(rel.x * c - rel.z * s, rel.y, rel.x * s + rel.z * c);
    samplePos = spun * scale + vec3<f32>(t * 0.14, t * 0.11, -t * 0.12);
  } else {
    // Planar sky cloud slab
    if (worldPos.y < minAlt || worldPos.y > maxAlt) {
      return 0.0;
    }
    heightFrac = (worldPos.y - minAlt) / (maxAlt - minAlt);
    samplePos = worldPos * scale + vec3<f32>(t, 0.0, t * 0.45);
  }

  // Rounded vertical cloud profile (thick middle, soft bottom & top)
  let profile = smoothstep(0.0, 0.22, heightFrac) * (1.0 - smoothstep(0.55, 1.0, heightFrac));
  if (profile <= 0.001) {
    return 0.0;
  }

  // Dynamic weather lifecycle: isotropic 3D domain-warped advection + buildup/dissipation + wispy edge erosion
  let wx = fbm3(samplePos * 0.52 + vec3<f32>(t * 0.14, -t * 0.11, t * 0.15)) - 0.47;
  let wz = fbm3(samplePos * 0.52 + vec3<f32>(-t * 0.12 + 5.3, t * 0.13 - 2.7, t * 0.10 + 1.9)) - 0.47;
  let warpedPos = samplePos + vec3<f32>(wx, (wx - wz) * 0.5, wz) * 0.95;
  let baseNoise = fbm3(warpedPos);
  let lifecycle = fbm3(samplePos * 0.40 + vec3<f32>(-t * 0.20, t * 0.24, t * 0.18));
  let edgeErosion = fbm3(warpedPos * 2.25 - vec3<f32>(t * 0.26, 0.0, t * 0.22));

  let localCoverage = clamp(coverage + (lifecycle - 0.47) * 0.55, 0.05, 0.95);
  let threshold = 1.0 - localCoverage;
  let shaped = smoothstep(threshold, threshold + 0.30, baseNoise - (1.0 - edgeErosion) * 0.10);
  return shaped * profile * density;
}

fn intersectSphere(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
  let oc = ro - center;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - radius * radius;
  let disc = b * b - c;
  if (disc < 0.0) {
    return vec2<f32>(1e9, -1e9);
  }
  let sq = sqrt(disc);
  return vec2<f32>(-b - sq, -b + sq);
}

fn computeCloudShadow(hitWorld: vec3<f32>, sunDir: vec3<f32>) -> f32 {
  let minAlt = vol.cloudLayer.x;
  let maxAlt = max(vol.cloudLayer.y, minAlt + 0.05);
  var t0 = 0.0;
  var t1 = 0.0;

  if (vol.cloudParams.z > 0.5) {
    let rel = hitWorld - vol.cloudBounds.xyz;
    let r = length(rel);
    let rOuter = vol.cloudBounds.w + maxAlt;
    if (r >= rOuter || dot(rel, sunDir) <= 0.0) {
      return 1.0;
    }
    let rInner = vol.cloudBounds.w + minAlt;
    let hitInner = intersectSphere(hitWorld, sunDir, vol.cloudBounds.xyz, rInner);
    let hitOuter = intersectSphere(hitWorld, sunDir, vol.cloudBounds.xyz, rOuter);
    t0 = select(0.0, max(0.0, hitInner.y), r < rInner);
    t1 = max(t0, hitOuter.y);
  } else {
    if (hitWorld.y >= maxAlt || sunDir.y <= 0.02) {
      return 1.0;
    }
    t0 = max(0.0, (minAlt - hitWorld.y) / sunDir.y);
    t1 = min(t0 + 18.0, (maxAlt - hitWorld.y) / sunDir.y);
  }

  let span = t1 - t0;
  if (span <= 0.005) {
    return 1.0;
  }

  let steps = 4;
  let stepLen = span / f32(steps);
  var opticalDepth = 0.0;
  for (var i = 0; i < steps; i++) {
    let t = t0 + (f32(i) + 0.5) * stepLen;
    opticalDepth += sampleCloudDensity(hitWorld + sunDir * t) * stepLen;
  }
  return exp(-opticalDepth * 1.85);
}

@fragment
fn fs_volumetric(in: VsOut) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(in.fragPos.xy);
  let baseColor = textureLoad(sceneColorTex, coord, 0);
  let ndcDepth = textureLoad(sceneDepthTex, coord, 0);

  let ndcXY = vec2<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
  let farClip = vol.invViewProj * vec4<f32>(ndcXY, 1.0, 1.0);
  let farWorld = farClip.xyz / max(abs(farClip.w), 1e-5);

  let ro = vol.cameraPos.xyz;
  let rd = normalize(farWorld - ro);

  var maxDist = 48.0;
  var hitWorld = farWorld;
  let hasHit = ndcDepth < 0.99995;
  if (hasHit) {
    let hitClip = vol.invViewProj * vec4<f32>(ndcXY, ndcDepth, 1.0);
    hitWorld = hitClip.xyz / max(abs(hitClip.w), 1e-5);
    maxDist = min(length(hitWorld - ro), 60.0);
  }

  let sunDir = normalize(vol.sunDir.xyz);
  let cosTheta = dot(rd, sunDir);
  let phase = mix(0.25, henyeyGreenstein(cosTheta, clamp(vol.cloudLayer.w, 0.05, 0.92)), 0.72);
  let sunTint = vec3<f32>(1.0, 0.95, 0.86);

  var transmittance = 1.0;
  var scattered = vec3<f32>(0.0);

  // 1. Exponential Height Fog
  let fogDensity = max(vol.sunDir.w, 0.0);
  if (fogDensity > 0.0001) {
    let fogH = max(vol.fogColor.w, 0.5);
    let avgY = ro.y + rd.y * (maxDist * 0.5);
    let heightFactor = exp(-max(avgY, 0.0) / fogH);
    let fogAmount = 1.0 - exp(-maxDist * fogDensity * heightFactor);
    let fogScatter = vol.fogColor.rgb + sunTint * (0.22 * pow(max(cosTheta, 0.0), 4.0));
    scattered += fogScatter * fogAmount;
    transmittance *= (1.0 - clamp(fogAmount, 0.0, 0.96));
  }

  // 2. Raymarched 3D Volumetric Clouds
  let cloudDensity = vol.cloudParams.x;
  if (cloudDensity > 0.0001) {
    var tStart = 0.0;
    var tEnd = maxDist;

    if (vol.cloudParams.z > 0.5) {
      // Spherical shell bounds for fast raymarching
      let rOuter = vol.cloudBounds.w + max(vol.cloudLayer.y, vol.cloudLayer.x + 0.05);
      let hitOuter = intersectSphere(ro, rd, vol.cloudBounds.xyz, rOuter);
      if (hitOuter.y > 0.0 && hitOuter.x < maxDist) {
        tStart = max(0.0, hitOuter.x);
        tEnd = min(maxDist, hitOuter.y);
      } else {
        tStart = 1.0;
        tEnd = 0.0;
      }
    } else {
      // Planar slab bounds
      let minY = vol.cloudLayer.x;
      let maxY = max(vol.cloudLayer.y, minY + 0.05);
      if (abs(rd.y) > 0.001) {
        let t0 = (minY - ro.y) / rd.y;
        let t1 = (maxY - ro.y) / rd.y;
        let tNear = min(t0, t1);
        let tFar = max(t0, t1);
        tStart = max(0.0, tNear);
        tEnd = min(maxDist, tFar);
      } else if (ro.y < minY || ro.y > maxY) {
        tStart = 1.0;
        tEnd = 0.0;
      }
    }

    let marchLen = tEnd - tStart;
    if (marchLen > 0.01) {
      let steps = 28;
      let stepSize = marchLen / f32(steps);
      // Interleaved gradient noise jitter to eliminate banding
      let jitter = fract(52.9829189 * fract(dot(in.fragPos.xy, vec2<f32>(0.06711056, 0.00583715))));
      var t = tStart + jitter * stepSize;

      for (var i = 0; i < steps; i++) {
        if (transmittance < 0.02) {
          break;
        }
        let pos = ro + rd * t;
        let d = sampleCloudDensity(pos);
        if (d > 0.001) {
          // 2-tap directional sun-shadowing inside the cloud
          let shadowStep = stepSize * 1.6;
          let shadowDensity =
            sampleCloudDensity(pos + sunDir * shadowStep) +
            sampleCloudDensity(pos + sunDir * shadowStep * 2.2) * 0.65;
          let lightTrans = exp(-shadowDensity * shadowStep * 1.45);
          let powder = 1.0 - exp(-d * stepSize * 2.2);
          let sunEnergy = mix(0.25, 1.0, lightTrans) * (0.65 + 0.75 * phase) * mix(0.55, 1.25, powder);

          // Day/night terminator modulation for spherical planetary clouds
          var planetDay = 1.0;
          if (vol.cloudParams.z > 0.5) {
            let pNorm = normalize(pos - vol.cloudBounds.xyz);
            planetDay = smoothstep(-0.22, 0.25, dot(pNorm, sunDir));
          }

          let cloudAmbient = vol.fogColor.rgb * 0.28 * mix(0.15, 1.0, planetDay);
          let cloudLit = sunTint * (sunEnergy * 1.35 * planetDay) + cloudAmbient;

          let stepTrans = exp(-d * stepSize * 1.6);
          scattered += transmittance * cloudLit * (1.0 - stepTrans);
          transmittance *= stepTrans;
        }
        t += stepSize;
      }
    }
  }

  let cloudShadow = select(1.0, computeCloudShadow(hitWorld, sunDir), hasHit && cloudDensity > 0.0001);
  let shadowedBase = baseColor.rgb * mix(0.36, 1.0, cloudShadow);

  let finalRgb = shadowedBase * transmittance + scattered;
  return vec4<f32>(finalRgb, baseColor.a);
}

