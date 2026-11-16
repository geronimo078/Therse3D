// Lightweight FXAA Edge Anti-Aliasing Post-Pass (ideal for Low-Spec 1x MSAA)

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fxaa(@builtin(vertex_index) vid: u32) -> VsOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  let p = positions[vid];
  var out: VsOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

fn luma(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_fxaa(in: VsOut) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(srcTex));
  let texel = 1.0 / max(dims, vec2<f32>(1.0));

  let rgbM = textureSample(srcTex, srcSampler, in.uv).rgba;
  let rgbNW = textureSample(srcTex, srcSampler, in.uv + vec2<f32>(-1.0, -1.0) * texel).rgb;
  let rgbNE = textureSample(srcTex, srcSampler, in.uv + vec2<f32>(1.0, -1.0) * texel).rgb;
  let rgbSW = textureSample(srcTex, srcSampler, in.uv + vec2<f32>(-1.0, 1.0) * texel).rgb;
  let rgbSE = textureSample(srcTex, srcSampler, in.uv + vec2<f32>(1.0, 1.0) * texel).rgb;

  let lumaM = luma(rgbM.rgb);
  let lumaNW = luma(rgbNW);
  let lumaNE = luma(rgbNE);
  let lumaSW = luma(rgbSW);
  let lumaSE = luma(rgbSE);

  let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
  let lumaRange = lumaMax - lumaMin;

  if (lumaRange < max(0.0312, lumaMax * 0.125)) {
    return rgbM;
  }

  var dir = vec2<f32>(
    -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
    (lumaNW + lumaSW) - (lumaNE + lumaSE),
  );
  let dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * 0.125), 1.0 / 128.0);
  let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2<f32>(-8.0), vec2<f32>(8.0)) * texel;

  let rgbA = 0.5 * (
    textureSample(srcTex, srcSampler, in.uv + dir * (1.0 / 3.0 - 0.5)).rgb +
    textureSample(srcTex, srcSampler, in.uv + dir * (2.0 / 3.0 - 0.5)).rgb
  );
  let rgbB = rgbA * 0.5 + 0.25 * (
    textureSample(srcTex, srcSampler, in.uv + dir * -0.5).rgb +
    textureSample(srcTex, srcSampler, in.uv + dir * 0.5).rgb
  );
  let lumaB = luma(rgbB);
  if (lumaB < lumaMin || lumaB > lumaMax) {
    return vec4<f32>(rgbA, rgbM.a);
  }
  return vec4<f32>(rgbB, rgbM.a);
}
