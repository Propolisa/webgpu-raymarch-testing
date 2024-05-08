@group(0) @binding(0) var<uniform> resolution: vec2f;
@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> mouse: vec2f;
@group(0) @binding(3) var<uniform> zoom: f32;
@group(0) @binding(4) var<uniform> upos: vec3f;
@group(0) @binding(5) var<uniform> urot: vec2f;



struct VertexInput {
    @location(0) pos: vec2f,
};

struct VertexOutput {
    @builtin(position) pos: vec4f,
};

@vertex
fn vertexMain(input: VertexInput) ->
    VertexOutput {
    var output: VertexOutput;
    output.pos = vec4f(input.pos, 0, 1);
    return output;
}

// Ray marching constants

const OCTAVES: i32 = 3; // {Octave count:0-8}
const CAVE_SIZE: f32 = 130.0; // {Entry room size:0-250}
const CAVE_THRESHOLD: f32 = 3.0;  // {How much to smooth the noise intersections:0-100}
const MAX_STEPS: i32  = 40; // {Max raymarch steps:0-100}
const NEAR_CLIP: f32 = 20.0; // {Near clipping distance: 0-100}
const SURF_DIST: f32 = 0.001; // {Max raymarch steps:0-1}
const MAX_DIST: f32 = 300.0;
const PI: f32 = 3.141592653592;
const TAU: f32 = 6.283185307185;
const DUST_RADIUS: f32 = 0.1; // {Dust radii:0-1}

// allow dynamic C.OCTAVES

const MAX_OCTAVES: i32 = 5;
const MIN_OCTAVES: i32 = 1;
const OCTAVES_FALLOFF_DISTANCE: f32 = C.MAX_DIST;


////////////////////////////////////////////////////////////////
// PBR Helper functions
////////////////////////////////////////////////////////////////

fn DistributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a      = roughness*roughness;
    let a2     = a*a;
    let NdotH  = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;
    let num   = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = C.PI * denom * denom;
    return num / denom;
}

fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r*r) / 8.0;
    let num   = NdotV;
    let denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

fn GeometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2  = GeometrySchlickGGX(NdotV, roughness);
    let ggx1  = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
} 
////////////////////////////////////////////////////////////////
// Random & Noise
////////////////////////////////////////////////////////////////

fn simpleHash( p0: vec3f ) -> vec3f
// Adapted from iq: https://www.shadertoy.com/view/Xsl3Dl
{
	var p = vec3( dot(p0,vec3(127.1,311.7, 74.7)),
			  dot(p0,vec3(269.5,183.3,246.1)),
			  dot(p0,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

fn gradientNoise( p : vec3f ) -> f32
// Adapted from iq: https://www.shadertoy.com/view/Xsl3Dl
{
    var i = floor( p );
    var f = fract( p );
    // cubic interpolant
    var u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( simpleHash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( simpleHash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( simpleHash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( simpleHash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( simpleHash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( simpleHash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( simpleHash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( simpleHash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

////////////////////////////////////////////////////////////////
// Transformations
////////////////////////////////////////////////////////////////

fn Rot(a: f32) -> mat2x2f {
    let s = sin(a);
    let c = cos(a);
    return mat2x2f(c, -s, s, c);
}

fn rotX(p: vec3f, a: f32) -> vec3f {
    let s = sin(a);
    let c = cos(a);
    let m = mat3x3f(
        1., 0., 0.,
        0., c, -s,
        0., s, c,
        );
    return m * p;
}

fn rotY(p: vec3f, a: f32) -> vec3f {
    let s = sin(a);
    let c = cos(a);
    let m = mat3x3f(
        c, 0., s,
        0., 1., 0.,
        -s, 0., c,
        );
    return m * p;
}

fn rotZ(p: vec3f, a: f32) -> vec3f {
    let s = sin(a);
    let c = cos(a);
    let m = mat3x3f(
        c, -s, 0.,
        s,  c, 0.,
        0., 0., 1.
        );
    return m * p;
}
////////////////////////////////////////////////////////////////
// SDF Operations
////////////////////////////////////////////////////////////////

fn opUnion(d1: f32, d2: f32 ) -> f32 { return min(d1,d2); }

fn opSubtraction(d1: f32, d2: f32) -> f32 {
    //NOTE: Flipped order because it makes more sense to me
    return max(-d2, d1);
}
fn opIntersection(d1: f32, d2: f32) -> f32 {
    return max(d1, d2);
}

fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}
fn opSmoothSubtraction(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d1, -d2, h ) + k*h*(1.0-h);
}

fn modDomain(p: vec3f, size: vec3f) -> vec3f {
    return p - size * round(p / size);
}

////////////////////////////////////////////////////////////////
// Signed Distance Functions
////////////////////////////////////////////////////////////////

fn sdPyramid(p: vec3<f32>, h: f32) -> f32 {
    let m2: f32 = h * h + 0.25;
    
    var p_mod: vec3<f32> = vec3<f32>(abs(p.x), p.y, abs(p.z));
    if (p_mod.z > p_mod.x) {
        p_mod.x = p_mod.z;
        p_mod.z = abs(p.x);  // Ensuring that p.zx swizzling is mirrored accurately
    }
    p_mod.x -= 0.5;
    p_mod.z -= 0.5;

    let q: vec3<f32> = vec3<f32>(p_mod.z, h * p_mod.y - 0.5 * p_mod.x, h * p_mod.x + 0.5 * p_mod.y);

    let s: f32 = max(-q.x, 0.0);
    let t: f32 = clamp((q.y - 0.5 * p_mod.z) / (m2 + 0.25), 0.0, 1.0);

    let a: f32 = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b: f32 = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

    var d2: f32 = 0.;
    if (min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) {
        d2 = 0.0;
    } else {
        d2 = min(a, b);
    }

    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}

fn sdPlane( p: vec3f, n: vec3f, h: f32 ) -> f32
{
  return dot(p,normalize(n)) + h;
}

fn sdSphere(p: vec3f, c: vec3f, r: f32) -> f32
{
    return length(p-c) - r;
}

fn sdRoundBox( po: vec3f, c: vec3f, b: vec3f, r: f32 ) -> f32
{
    let p = po - c;
    let q = abs(p) - b;
    return length(max(q,vec3f(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}


////////////////////////////////////////////////////////////////
// Main scene
////////////////////////////////////////////////////////////////

fn orbitControls(po: vec3f) -> vec3f {
    let m = (vec2f(C.mouse.x, C.mouse.y) / C.resolution) + 0.5;
    var p = po;
    p = rotY(po, -m.x*C.TAU);
    p = rotX(p, -m.y*C.PI);
    return p;
}

fn worleyNoise(p: vec3f, seed: vec3f) -> vec2f {
    var d = vec2f(1e10, 1e10); // Store two smallest distances
    var st = floor(p);
    for (var i: i32 = -1; i <= 1; i += 2) {
        for (var j: i32 = -1; j <= 1; j += 2) {
            for (var k: i32 = -1; k <= 1; k += 2) {
                var g = st + vec3f(f32(i), f32(j), f32(k));
                var o = simpleHash(g + seed); // Using simpleHash from your existing code
                var r = g + o - p;
                var t = dot(r, r);  // Use squared distance for comparison

                if (t < d.x) {
                    d.y = d.x;
                    d.x = t;
                } else if (t < d.y) {
                    d.y = t;
                }
            }
        }
    }
    d.x = sqrt(d.x); // Now compute the square root for the two nearest distances
    d.y = sqrt(d.y);
    return d;
}

fn sdWorleyNoise(p: vec3f, seed: vec3f) -> f32 {
    let d = worleyNoise(p, seed);
    return d.y - d.x; // SDF based on difference between closest and second closest
}


fn mod7v3f(x: vec3f) -> vec3f { return x - floor(x / 6.999999) * 6.999999; }
fn mod7v4f(x: vec4f) -> vec4f { return x - floor(x / 6.999999) * 6.999999; }

  // Special thanks to Stefan Gustavson for releasing mod289 as public domain code!
  // Always credit the original author to show appreciation.
fn mod289f32(x: f32)   ->   f32 { return x - floor(x / 289.0) * 289.0; }
fn mod289v2f(x: vec2f) -> vec2f { return x - floor(x / 289.0) * 289.0; }
fn mod289v3f(x: vec3f) -> vec3f { return x - floor(x / 289.0) * 289.0; }
fn mod289v4f(x: vec4f) -> vec4f { return x - floor(x / 289.0) * 289.0; }

fn permute289f32(x:   f32) ->   f32 { return mod289f32(((x*34.0) + 10.0) * x); }
fn permute289v3f(x: vec3f) -> vec3f { return mod289v3f((34.0 * x + 10.0) * x); }
fn permute289v4f(x: vec4f) -> vec4f { return mod289v4f((34.0 * x + 10.0) * x); }

  // These fade functions have been separated from Stefan Gustavson's cnoise functions:
  // - fadev2f separated from the cnoise2D file
  // - fadev3f separated from the cnoise3D file
  // - fadev4f separated from the cnoise4D file
fn fadev2f(t: vec2f) -> vec2f { return t*t*t*(t*(t*6.0 - 15.0) + 10.0); }
fn fadev3f(t: vec3f) -> vec3f { return t*t*t*(t*(t*6.0 - 15.0) + 10.0); }
fn fadev4f(t: vec4f) -> vec4f { return t*t*t*(t*(t*6.0 - 15.0) + 10.0); }

fn taylorInvSqrtf32(r:   f32) ->   f32 { return 1.79284291400159 - 0.85373472095314 * r; }
fn taylorInvSqrtv4f(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }

fn snoise3D(v: vec3f) -> f32 {
    let C = vec2f(1./6., 1./3.);
    let D = vec4f(0., .5, 1., 2.);

    var i = floor(v + dot(v, C.yyy));
    var x0 = v - i + dot(i, C.xxx);

    var g = step(x0.yzx, x0.xyz);
    var l = 1.0 - g;
    var i1 = min( g.xyz, l.zxy );
    var i2 = max( g.xyz, l.zxy );

    var x1 = x0 - i1 + C.xxx;
    var x2 = x0 - i2 + C.yyy;
    var x3 = x0 - D.yyy;

    i = mod289v3f(i);
    var p = permute289v4f( permute289v4f( permute289v4f(
              i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    var n_ = 0.142857142857;
    var ns = n_ * D.wyz - D.xzx;

    var j = p - 49.0 * floor(p * ns.z * ns.z);

    var x_ = floor(j * ns.z);
    var y_ = floor(j - 7.0 * x_ );

    var x = x_ *ns.x + ns.yyyy;
    var y = y_ *ns.x + ns.yyyy;
    var h = 1.0 - abs(x) - abs(y);

    var b0 = vec4f( x.xy, y.xy );
    var b1 = vec4f( x.zw, y.zw );

    var s0 = floor(b0)*2.0 + 1.0;
    var s1 = floor(b1)*2.0 + 1.0;
    var sh = -step(h, vec4(0.0));

    var a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    var a1 = b1.xzyw + s1.xzyw*sh.zzww;

    var p0 = vec3f( a0.xy, h.x );
    var p1 = vec3f( a0.zw, h.y );
    var p2 = vec3f( a1.xy, h.z );
    var p3 = vec3f( a1.zw, h.w );

    var norm = taylorInvSqrtv4f( vec4f( dot( p0, p0 ), dot( p1, p1 ), dot( p2, p2 ), dot( p3, p3 ) ));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    var m = max(0.5 - vec4f( dot( x0, x0 ), dot( x1, x1 ), dot( x2, x2 ), dot( x3, x3 ) ), vec4f(0.0));
    m = m * m;

    return 105.0 * dot( m*m, vec4f( dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3) ));
  }


fn fBm(p: vec3f) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.9;
    var frequency: f32 = 0.5;
    
    for (var i: i32 = 0; i < C.OCTAVES; i++) {
        value += (f32(i) * 0.001) + (amplitude * snoise3D(p * frequency));
        frequency *= (f32(i) * 0.05) + 2.0;
        amplitude *= (f32(i) * 0.05) + 0.5;
    }
    
    return value;
}

fn pseudoRandom(seed: i32) -> f32 {
    var x = seed * 374761393;
    x = (x ^ (x >> 13)) * 1103515245;
    x = x ^ (x >> 16);
    return fract(f32(x) / 4294967296.0);
}



fn drawSnake(
    startPos: vec3<f32>, 
    direction: vec3<f32>, 
    length: f32, 
    segmentCount: i32, 
    segmentRadius: f32, 
    originalValue: f32, 
    speed: f32, 
    wriggleAmplitude: f32
) -> f32 {
    var returnable = originalValue;
    var i: i32 = 0;
    let loopPeriod = 3.0;
    let timeModulus = fract(C.time / loopPeriod);

    // Calculate wriggle speed based on global positional speed
    let wriggleSpeed = speed * 1.0; // Wriggle frequency scaled with speed

    loop {
        if i >= segmentCount { break; }

        let wrigglePhase = -2.0 * 3.14159265 * (wriggleSpeed * timeModulus + f32(i) * 0.3);
        let wriggleFactor = sin(wrigglePhase) * wriggleAmplitude;
        
        let movementVector = direction * (speed * C.time - (length / f32(segmentCount) * f32(i)));
        let segmentX = startPos.x + wriggleFactor + movementVector.x;
        let segmentY = startPos.y + wriggleFactor + movementVector.y;
        let segmentZ = startPos.z + movementVector.z;

        let segmentPos = vec3<f32>(segmentX, segmentY, segmentZ);
        returnable = opSmoothUnion(sdSphere(segmentPos, vec3f(0.0), segmentRadius), returnable, 0.7);
        i++;
    }

    return returnable;
}

fn customNoise3D(p: vec3f, amplitude: f32, scale: f32, octaves: i32) -> f32 {
    var value: f32 = 0.0;
    var currentAmplitude: f32 = amplitude;
    var currentScale: f32 = scale;

    for (var i: i32 = 0; i < octaves; i++) {
        value += currentAmplitude * snoise3D(p * currentScale);
        currentScale *= 2.0;   // Increasing frequency
        currentAmplitude *= 0.5; // Decreasing amplitude
    }

    return value;
}

fn hash3(seed: f32) -> vec3f {
    var p = vec3f(sin(seed * 91.3458), sin(seed * 47.9897), sin(seed * 74.233));
    return fract(p * 43758.5453);
}

fn hash(p: u32) -> f32 {
    var x = p * 374761393u;
    x = (x ^ (x >> 13u)) * 1103515245u;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967296.0;
}

fn mainScene(p: vec3f) -> f32 {
    let distToCamera = length(p - C.upos);
    let normalizedDistance = clamp(distToCamera / C.OCTAVES_FALLOFF_DISTANCE, 0.0, 1.0);
    // let octaveCount = i32(mix(f32(MAX_CELO.OCTAVES), f32(MIN_CELO.OCTAVES), normalizedDistance));
    let octaveCount = C.OCTAVES;

    let largeScaleNoise = customNoise3D(p, 1.0, 0.05, octaveCount);
    let largerScaleNoise = customNoise3D(p, 6.0, 0.01, octaveCount + 1 );
    let solidVolumeSDF = sdSphere(p, vec3f(0.0), C.CAVE_SIZE);
    let caveSDF = opSmoothSubtraction(opSmoothSubtraction(largeScaleNoise, solidVolumeSDF, C.CAVE_THRESHOLD), largerScaleNoise, C.CAVE_THRESHOLD/10);

    var returnable = caveSDF;
    let numSnakes = 0; // Number of snakes to generate

    for (var j: i32 = 0; j < numSnakes; j++) {
        let seedBase = j * 137;
        let randomDirection = normalize(vec3<f32>(
            pseudoRandom(seedBase + 1) - 0.5,
            pseudoRandom(seedBase + 2) - 0.5,
            pseudoRandom(seedBase + 3) - 0.5
        ));
        let randomSpeed = 0.5 + 12 * pseudoRandom(seedBase + 4); // Increased speed range
        let randomStartPosition = p + 20.0 * randomDirection;
        let randomLength = 5.0 + 10.0 * pseudoRandom(seedBase + 5);
        let randomSegmentCount = i32(randomLength) + 2;
        let randomSegmentRadius = 0.2 + 1.80 * pseudoRandom(seedBase + 7);
        let randomWriggleAmplitude = 0.1 + 2.4 * pseudoRandom(seedBase + 8); // Amplitude between 0.1 and 0.5

        returnable = drawSnake(
            randomStartPosition,
            randomDirection,
            randomLength,
            randomSegmentCount,
            randomSegmentRadius,
            returnable,
            randomSpeed,
            randomWriggleAmplitude
        );
    }

    return returnable;
//   let id = vec3<u32>(
//         u32(floor((p.x + 2.0) / 4.0)),
//         u32(floor((p.y + 2.0) / 4.0)),
//         u32(floor((p.z + 2.0) / 4.0))
//     );
//     let localPos =  modDomain(p + vec3f(2.0), vec3f(4.0)) - vec3f(2.0);
//     var minDist = 1e10;

//     for (var i: u32 = 0u; i < 4u; i++) {
//         var hashOffset = simpleHash(vec3<f32>(f32(i), f32(id.x), f32(id.y)));
//         var offset = vec3<f32>(1.7, 3.2, 1.7) * hashOffset;
//         offset += vec3<f32>(0.3, 0.15, 0.3) * sin(0.3 * C.time + vec3<f32>(f32(i + id.x), f32(i + 3u + id.y), f32(i * 2u + 1u + 2u * id.x)));
//         let dist = length(localPos - offset);
//         minDist = min(dist * dist, minDist);
//     }
//     return min(sqrt(minDist) -  0.02, fBm(p));  // Increase the subtracted value to effectively increase the radius of influence
    // return step(0.5, customNoise3D(p, 0.000001, 0.05, C.OCTAVES));
    // let time = C.time;
    // let scale = 1000.0; // Control the scale of the grid
    // let localPos = modDomain(p, vec3f(scale)); // Wrap domain around scale
    // let id = vec3f(
    //     localPos.x,
    //    localPos.y,
    //     localPos.z,
    // );
    // let t = time / 4.0;
    // let bias = vec3f(0.0, 4.0 * sin(f32(id.x) * 128.0 + t), 0.0);
    // let st = fract(localPos + bias); // Apply bias based on time and x-position

    // let mask = smoothstep(0.1, 0.2, -cos(f32(id.x) * 128.0 + t));
    // let noiseValue = customNoise3D(vec3f(id.x, id.y, t), 1.0, 0.05, C.OCTAVES); // Simulate a noise function
    // let size = noiseValue * 50 + 0.01;
    // let pos = vec3f(customNoise3D(vec3f(t, f32(id.y) * 64.1, 0.0), 1.0, 0.05, C.OCTAVES) * 0.8 + 0.1, 0.5, 0.0);

    // if (length(st.xy - pos.xy) < size) {
    //     let sphereSDF = sdSphere(localPos, pos, size);
    //     return -sphereSDF * mask; // Invert and apply mask for bubble effect
    // }
    // return 1.0; // Non-bubble space

}





fn getDist(p: vec3f) -> f32 {
    return mainScene(p); // Retrieve the scene's SDF
}

fn getNormal(p: vec3f) -> vec3f {
    let epsilon = 0.0001;
    let dx = vec3(epsilon, 0., 0.0);
    let dy = vec3(0., epsilon, 0.0);
    let dz = vec3(0., 0.0, epsilon);

    let ddx = getDist(p + dx) - getDist(p - dx);
    let ddy = getDist(p + dy) - getDist(p - dy);
    let ddz = getDist(p + dz) - getDist(p - dz);
    
    return normalize(vec3f(ddx, ddy, ddz));
}

// COORDINATE SYSTEM: X = [-1,+1] (Right pos) | Y = [-1,+1] (Down pos.)  

////////////////////////////////////////////////////////////////
// Ray Marching Functions
////////////////////////////////////////////////////////////////

fn rayDirection(p: vec2f, ro: vec3f, rt: vec3f, up: vec3f, yaw: f32, pitch: f32, fov: f32, aspectRatio: f32) -> vec3f {
    let cosPitch = cos(pitch);
    let sinPitch = sin(pitch);
    let cosYaw = cos(yaw);
    let sinYaw = sin(yaw);

    // Forward direction vector based on yaw and pitch (Euler angles)
    let forward = vec3f(cosPitch * sinYaw, sinPitch, cosPitch * cosYaw);

    // Right vector perpendicular to 'forward' and 'up'
    let right = normalize(cross(up, forward));

    // Adjust 'up' vector to be perpendicular to 'forward' and 'right'
    let adjustedUp = cross(forward, right);

    // Calculate normalized device coordinates (NDC) offset for field of view
    let fovTan = tan(fov / 2.0);
    let ndcX = p.x * fovTan * aspectRatio;
    let ndcY = p.y * fovTan;

    // Calculate the final ray direction
    return normalize(forward + ndcX * right + ndcY * adjustedUp);
}




fn rayMarch(ro: vec3f, rd: vec3f) -> f32 {
    var d = C.NEAR_CLIP;
    var i: i32 = 0;
    loop {
        if i >= C.MAX_STEPS { break; }
        let p = ro + rd * d;
        let ds = getDist(p);
        d += ds;
        if d >= C.MAX_DIST || ds < C.SURF_DIST { break; }
        i++;
    }
    return d;
}


////////////////////////////////////////////////////////////////
// Scene constants
////////////////////////////////////////////////////////////////

const numLights = 1;
const baseLightPower = 18.0;
const lights = array<vec3f, numLights>(
    vec3f(4.0, -2.0, -4.0),
    // vec3f(-1, -.25, 1.),
    // vec3f(0., -10.0, 0.),
    // vec3f(0., 20.0, 0.)
);
const lightPowers = array<f32, numLights>( 4.0 /**, 1.0, 2.0, 1.0 */);
const lightColors = array<vec3f, numLights>(
    vec3f(1.0, 0.9, 0.9),
    // vec3f(1.0, 1.0, 0.0),
    // vec3f(1.0, 1.0, 0.0),
    // vec3f(1.0, 1.0, 0.0),
);

@fragment
fn fragmentMain(@builtin(position) pos: vec4<f32>) -> @location(0) vec4f {
    let uv = (vec2(pos.x, pos.y) / C.resolution - 0.5) * 2.0; // Normalize screen coordinates

    // Camera and ray setup
    let ro = C.upos;  // Camera position updated from uniforms
    let rt = vec3f(0.0, 0.0, 0.0); // World origin as the target position
    let up = vec3f(0.0, 1.0, 0.0); // World up vector
    let fovRadians = radians(50.0 * C.zoom); // Field of view adjusted by C.zoom
    let aspectRatio = C.resolution.x / C.resolution.y;

    // Calculate ray direction and perform ray marching
    let rd = rayDirection(uv, ro, rt, up, C.urot.y, C.urot.x, fovRadians, aspectRatio);
    let d = rayMarch(ro, rd);
    // return vec4(vec3f(d / C.MAX_DIST), 1.0);
    // Background gradient computation
    var v = length(uv) * 0.75;
    var fragColor = vec4f(mix(0.1, 0.2, smoothstep(0.0, 1.0, uv.y)));


    if (d <= C.MAX_DIST) {
        let p = ro + rd * d;
        let N = getNormal(p);
        let V = -rd;

        // PBR shading parameters
        let albedo = vec3f(1.0, 1.0, 0.0);
        let roughness = 0.8;
        let metallic = 0.0;
        var F0 = vec3(0.04);
        F0 = mix(F0, albedo, metallic);

        // Lighting calculation
        var Lo = vec3f(0.0);
        for (var i: i32 = 0; i < numLights; i++) {
            let lightPos = lights[i] + C.upos;
            let L = normalize(lightPos - p);
            let H = normalize(V + L);
            let radiance = lightColors[i] * (1.0 / (length(lightPos - p) * length(lightPos - p)));
            let NDF = DistributionGGX(N, H, roughness);
            let G = GeometrySmith(N, V, L, roughness);
            let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
            let specular = NDF * G * F / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
            Lo += (vec3f(1.0) - F + specular) * radiance * max(dot(N, L), 0.0) * baseLightPower * lightPowers[i];
        }

        let ambient = vec3f(0.01) * albedo;
        var color = ambient + Lo;
        color = color / (color + vec3f(1.0));  // Tone mapping
        color = pow(color, vec3f(1.0/2.2));  // Gamma correction
        fragColor = vec4(color, 1.0);

        // Modify the output color based on distance to make distant objects bluer
        let depthColor = vec4f(0.0, 0.1, .9, 1.0); // Pure blue at maximum distance
        let blueFactor = d / C.MAX_DIST; // Normalize distance to range [0, 1]
        fragColor = mix(fragColor, depthColor, blueFactor); // Blend original color with blue based on distance
    }

    return fragColor;
}


// @fragment
// fn fragmentMain(@builtin(position) pos: vec4<f32>) -> @location(0) vec4f {
//     let uv = (vec2(pos.x, pos.y) / C.resolution - 0.5) * 2.0; // Normalize screen coordinates

//     // Camera and ray setup
//     let ro = C.upos;  // Camera position updated from uniforms
//     let rt = vec3f(0.0, 0.0, 0.0); // World origin as the target position
//     let up = vec3f(0.0, 1.0, 0.0); // World up vector
//     let fovRadians = radians(50.0 * C.zoom); // Field of view adjusted by C.zoom
//     let aspectRatio = C.resolution.x / C.resolution.y;

//     // Calculate ray direction and perform ray marching
//     let rd = rayDirection(uv, ro, rt, up, C.urot.y, C.urot.x, fovRadians, aspectRatio);
//     let d = rayMarch(ro, rd);

//     // Background gradient computation
//     var v = length(uv) * 0.75;
//     var fragColor = vec4f(mix(0.1, 0.2, smoothstep(0.0, 1.0, uv.y)));
//     fragColor += mix(vec4f(0.6, 0.6, 0.9, 1.0), vec4f(0.0, 0.0, 0.0, 1.0), v);  // Adjusted to a cooler tone

//     if (d <= 100.0) {
//         let p = ro + rd * d;

//         // Modify the output color based on distance traveled by the ray
//         let distanceFactor = clamp(d / 100.0, 0.0, 1.0); // Normalize distance to range [0, 1]
//         let grayscale = vec3f(distanceFactor); // Convert distance factor to grayscale color
//         let depthColor = vec4f(grayscale * vec3f(0.0, 0.0, 1.0), 1.0); // Multiply by blue color

//         // Blend the original color with the distance-based color
//         fragColor = mix(fragColor, depthColor, distanceFactor);
//     }

//     return fragColor;
// }
