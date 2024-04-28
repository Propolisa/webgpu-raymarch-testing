@group(0) @binding(0) var<uniform> rez: vec2f;
@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> mouse: vec2f;

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

const OCTAVES: i32 = 4;
const CAMERA_SPEED: f32 = 0.1; // The constant speed at which the camera moves
const CAVE_SIZE: f32 = 20.0;
const CAVE_THRESHOLD: f32 = 0.0;
const MAX_STEPS = 500;
const SURF_DIST = 0.001;
const MAX_DIST = 100.0;
const PI = 3.141592653592;
const TAU = 6.283185307185;

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
    denom = PI * denom * denom;
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

////////////////////////////////////////////////////////////////
// Signed Distance Functions
////////////////////////////////////////////////////////////////
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
    let m = (vec2f(mouse.x, mouse.y) / rez) + 0.5;
    var p = po;
    p = rotY(po, -m.x*TAU);
    p = rotX(p, -m.y*PI);
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
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    
    for (var i: i32 = 0; i < OCTAVES; i++) {
        value += amplitude * snoise3D(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    
    return value;
}


fn mainScene(p: vec3f) -> f32 {
    let largeScaleNoise = fBm(p * 0.1); // Large scale for the noise to affect the volume
    let solidVolumeSDF = -sdSphere(p, vec3f(0.0), CAVE_SIZE); // Negative SDF for the volume
    let caveSDF = solidVolumeSDF + (largeScaleNoise * 0.5); // Apply the noise influence

    // Perform a smooth subtraction of the noise from the volume
    let smoothness = 0.5; // Adjust the smoothness of the transition
    let threshold = 0.2; // Threshold for the cave's surface, tweak for different cave sizes
    let caveInterior = opSmoothSubtraction(caveSDF, threshold, smoothness);

    return caveInterior;
}

fn getDist(p: vec3f) -> f32 {
    let pRotated = rotX(rotY(p, -0.005 * time), 0.01 * time); // Apply rotation based on time
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

fn rayDirection(p: vec2f, ro: vec3f, rt: vec3f) -> vec3f {

    // screen orientation
    let vup = vec3f(0., 1.0, 0.0);
    let aspectRatio = rez.y / rez.x;

    let vw = normalize(ro - rt);
    let vu = normalize(cross(vup, vw));
    let vv = cross(vw, vu);
    let theta = radians(30.); // half FOV
    let viewport_height = 2. * tan(theta);
    let viewport_width = aspectRatio * viewport_height;
    let horizontal = -viewport_width * vu;
    let vertical = viewport_height * vv;
    let focus_dist = length(ro - rt);
    let center = ro - vw * focus_dist;

    let rd = center + p.x * horizontal + p.y * vertical - ro;

    return normalize(rd);
}

fn rayMarch(ro: vec3f, rd: vec3f) -> f32 {
    var d = 0.0;
    var i: i32 = 0;
    loop {
        if i >= MAX_STEPS { break; }
        let p = ro + rd * d;
        let ds = getDist(p);
        d += ds;
        if d >= MAX_DIST || ds < SURF_DIST { break; }
        i++;
    }
    return d;
}

////////////////////////////////////////////////////////////////
// Scene constants
////////////////////////////////////////////////////////////////

const numLights = 4;
const baseLightPower = 18.0;
const lights = array<vec3f, numLights>(
    vec3f(4.0, -2.0, -4.0),
    vec3f(-1, -.25, 1.),
    vec3f(0., -10.0, 0.),
    vec3f(0., 20.0, 0.)
);
const lightPowers = array<f32, numLights>( 4.0, 1.0, 2.0, 1.0 );
const lightColors = array<vec3f, numLights>(
    vec3f(1.0, 0.9, 0.9),
    vec3f(1.0),
    vec3f(0.9, 0.9, 1.0),
    vec3f(1.0),
);

const MAX_SPEED: f32 = 0.02; // Maximum speed of the camera

struct CameraState {
    position: vec3f,
    direction: vec3f
}

fn randomDirectionChange(time: f32) -> vec3f {
    // Generate a small random direction change based on time
    let randomHash = simpleHash(vec3f(time, time * 0.1, time * 0.2));
    // Normalize and scale to get a small direction change
    let smallChange = normalize(randomHash) * MAX_SPEED;
    return smallChange;
}

fn moveCamera(ro: vec3f, rd: vec3f, time: f32) -> CameraState {
    let dirChange = randomDirectionChange(time);
    let newDir = normalize(rd + dirChange); // Ensure the direction remains normalized
    let newPos = ro + newDir * MAX_SPEED; // Move the camera along the new direction at a constant speed
    return CameraState(newPos, newDir);
}

@fragment
fn fragmentMain(@builtin(position) pos: vec4<f32>) -> @location(0) vec4f {
    let uv = (vec2(pos.x, pos.y) / rez - 0.5) * 2.0;
    let initialRo = vec3f(0.0, 0.0, 5.0); // Initial camera position inside the cave
    let initialRd = vec3f(0.0, 0.0, -1.0); // Initial camera view direction (looking forward)

    // Animate the camera with random movement and view direction changes
    let cp = moveCamera(initialRo, initialRd, time);
    let ro = cp.position;
    let rd = cp.direction;

    // Perform ray marching with the updated camera position and direction
    let rayDir = rayDirection(uv, ro, ro + rd); // Target is the new camera direction
    let d = rayMarch(ro, rayDir);


    // Background
    var v = length(uv) * .75;
    var fragColor = vec4f(mix(0.1, 0.2, smoothstep(0.0, 1.0, uv.y)));
	fragColor += mix(vec4f(0.6), vec4f(0.0, 0.0, 0.0, 1.0), v);


    if (d <= 100.0) {
        let p = ro + rd * d;
        let N = getNormal(p);
        let V = -rd;

        // PBR Shading
        // material parameters
        let albedo = vec3f(1.0, 0.62, 0.26);
        let roughness = 0.15;
        let metallic = 0.0;
        var F0 = vec3(0.04);
        F0 = mix(F0, albedo, metallic);

        // calculate per-light radiance
        //WGSL
        var i = 0;
        var lightPos: vec3f;
        var Lo = vec3f(0.);
        loop {
            if i >= numLights { break; }
            lightPos = lights[i];

            let L = normalize(lightPos - p);
            let H = normalize(V + L);
            let distance    = length(lightPos - p);
            let attenuation = 1.0 / (distance * distance);
            let radiance    = lightColors[i] * attenuation;        
            
            // cook-torrance brdf
            let NDF = DistributionGGX(N, H, roughness);        
            let G   = GeometrySmith(N, V, L, roughness);      
            let F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
            
            let kS = F;
            var kD = vec3f(1.0) - kS;
            kD *= 1.0 - metallic;	  
            
            let numerator   = NDF * G * F;
            let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
            let specular    = numerator / denominator;  
                
            // add to outgoing radiance Lo
            let NdotL = max(dot(N, L), 0.0);                
            Lo += (kD * albedo / PI + specular) * radiance * NdotL * baseLightPower * lightPowers[i]; 
            i++;
        }
        let ambient = vec3f(0.01) * albedo;
        var color = ambient + Lo;
        
        // Gamma correction
        color = color / (color + vec3f(1.0));
        color = pow(color, vec3f(1.0/2.2));  
    
        fragColor = vec4(color, 1.0);
        
    }

    return fragColor;
} 