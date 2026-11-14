struct BloomUniforms {
  texelSize: vec2f,
  threshold: f32,
  intensity: f32,
  radius: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> uBloom: BloomUniforms;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var sceneSampler: sampler;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_bloom(@builtin(vertex_index) vid: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  let p = positions[vid];
  var out: VertexOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

fn extractBright(rgb: vec3f, threshold: f32) -> vec3f {
  let luma = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  let knee = 0.25;
  let soft = clamp((luma - threshold + knee) / (2.0 * knee), 0.0, 1.0);
  let weight = max(luma - threshold, 0.0) + soft * soft * knee;
  return rgb * (weight / max(luma, 1e-4));
}

@fragment
fn fs_bloom(in: VertexOut) -> @location(0) vec4f {
  let base = textureSampleLevel(sceneTex, sceneSampler, in.uv, 0.0).rgb;
  let r1 = uBloom.texelSize * (1.8 * uBloom.radius);
  let r2 = uBloom.texelSize * (4.5 * uBloom.radius);
  let thresh = uBloom.threshold;

  // 2-ring 12-tap + center bright-pass filter
  var glow = extractBright(base, thresh) * 0.16;

  // Inner ring (4 diagonal + 4 cardinal = 8 taps, weight 0.07 each = 0.56)
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( r1.x,  0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f(-r1.x,  0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( 0.0,  r1.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( 0.0, -r1.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( r1.x,  r1.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f(-r1.x,  r1.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( r1.x, -r1.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f(-r1.x, -r1.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.07;

  // Outer ring (4 cardinal + 4 diagonal = 8 taps, weight 0.035 each = 0.28)
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( r2.x,  0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f(-r2.x,  0.0), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( 0.0,  r2.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( 0.0, -r2.y), vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( r2.x,  r2.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f(-r2.x,  r2.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f( r2.x, -r2.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;
  glow += extractBright(textureSampleLevel(sceneTex, sceneSampler, clamp(in.uv + vec2f(-r2.x, -r2.y) * 0.707, vec2f(0.0), vec2f(1.0)), 0.0).rgb, thresh) * 0.035;

  let combined = base + glow * uBloom.intensity;
  return vec4f(combined, 1.0);
}
