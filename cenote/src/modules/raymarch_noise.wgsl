// @group(0) @binding(0) var<uniform> resolution: vec2f;
// @group(0) @binding(1) var<uniform> time: f32;
// @group(0) @binding(2) var<uniform> mouse: vec2f;
// @group(0) @binding(3) var<uniform> zoom: f32;
// @group(0) @binding(4) var<uniform> upos: vec3f;
// @group(0) @binding(5) var<uniform> urot: vec2f;


// psrdnoise3.wgsl

//
// Authors: Stefan Gustavson (stefan.gustavson@gmail.com)
// and Ian McEwan (ijm567@gmail.com)
// Version 2022-02-28, published under the MIT license (see below)
//
// Copyright (c) 2021-2022 Stefan Gustavson and Ian McEwan.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//

// WGSL lacks overloading of user defined functions, and considering
// the unfinished state of the platform I don't trust the dead code
// removal, so the functions are named after their argument lists:
//
//	psrnoise3(x: vec3<f32>, p: vec3<f32>, alpha: f32) -> f32
//	psnoise3(x: vec3<f32>, p: vec3<f32>) -> f32
//	srnoise3(x: vec3<f32>, alpha: f32) -> f32
//	snoise3(x: vec3<f32>) -> f32
//	psrdnoise3(x: vec3<f32>, p: vec3<f32>, alpha: f32) -> NG3
//	psdnoise3(x: vec3<f32>, p: vec3<f32>) -> NG3
//	srdnoise3(x: vec3<f32>, alpha: f32) -> NG3
//	sdnoise3(x: vec3<f32>) -> NG3
// The struct type NG3 is declared below.

// Struct for returning noise and its analytic gradient
struct NG3 {
	noise: f32,
	gradient: vec3<f32>
};

fn mod289v4f_psrn(i: vec4<f32>) -> vec4<f32> {
	return i - floor(i / 289.0) * 289.0;
}

fn permute289v4f_psrn(i: vec4<f32>) -> vec4<f32>
{
	var im: vec4<f32> = mod289v4f_psrn(i);
	return mod289v4f_psrn((im*34.0 + 10.0)*im);
}

fn psrnoise3(x: vec3<f32>, p: vec3<f32>, alpha: f32) -> f32
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
	
	var vx: vec4<f32>;
	var vy: vec4<f32>;
	var vz: vec4<f32>;

	if(any(p > vec3<f32>(0.0))) {
		vx = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
		vy = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
		vz = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
		if(p.x > 0.0) {
			vx = vx - floor(vx / p.x) * p.x;
		}
		if(p.y > 0.0) {
			vy = vy - floor(vy / p.y) * p.y;
		}
		if(p.z > 0.0) {
			vz = vz - floor(vz / p.z) * p.z;
		}
		i0 = floor(M * vec3<f32>(vx.x, vy.x, vz.x) + 0.5);
		i1 = floor(M * vec3<f32>(vx.y, vy.y, vz.y) + 0.5);
		i2 = floor(M * vec3<f32>(vx.z, vy.z, vz.z) + 0.5);
		i3 = floor(M * vec3<f32>(vx.w, vy.w, vz.w) + 0.5);
	}
	
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;
	var px: vec4<f32>;
	var py: vec4<f32>;
	var pz: vec4<f32>;
	var Sp: vec4<f32>;
	var Cp: vec4<f32>;
	var Ctp: vec4<f32>;
	var qx: vec4<f32>;
	var qy: vec4<f32>;
	var qz: vec4<f32>;
	var Sa: vec4<f32>;
	var Ca: vec4<f32>;

	if(alpha != 0.0)
	{
		px = Ct * sz_;
		py = St * sz_;
		pz = sz;
		Sp = sin(psi);
		Cp = cos(psi);
		Ctp = St*Sp - Ct*Cp;
		qx = mix( Ctp*St, Sp, sz);
		qy = mix(-Ctp*Ct, Cp, sz);
		qz = -(py*Cp + px*Sp);
		Sa = vec4<f32>(sin(alpha));
		Ca = vec4<f32>(cos(alpha));
		gx = Ca*px + Sa*qx;
		gy = Ca*py + Sa*qy;
		gz = Ca*pz + Sa*qz;
	}
	else
	{
		gx = Ct * sz_;
		gy = St * sz_;
		gz = sz;  
	}
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = dot(w3, gdotx);
	
	return 39.5 * n;
}

fn psnoise3(x: vec3<f32>, p: vec3<f32>) -> f32
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
	
	var vx: vec4<f32>;
	var vy: vec4<f32>;
	var vz: vec4<f32>;

	if(any(p > vec3<f32>(0.0))) {
		vx = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
		vy = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
		vz = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
		if(p.x > 0.0) {
			vx = vx - floor(vx / p.x) * p.x;
		}
		if(p.y > 0.0) {
			vy = vy - floor(vy / p.y) * p.y;
		}
		if(p.z > 0.0) {
			vz = vz - floor(vz / p.z) * p.z;
		}
		i0 = floor(M * vec3<f32>(vx.x, vy.x, vz.x) + 0.5);
		i1 = floor(M * vec3<f32>(vx.y, vy.y, vz.y) + 0.5);
		i2 = floor(M * vec3<f32>(vx.z, vy.z, vz.z) + 0.5);
		i3 = floor(M * vec3<f32>(vx.w, vy.w, vz.w) + 0.5);
	}
	
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;

	gx = Ct * sz_;
	gy = St * sz_;
	gz = sz;  
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = dot(w3, gdotx);
	
	return 39.5 * n;
}

fn srnoise3(x: vec3<f32>, alpha: f32) -> f32
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
	
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;
	var px: vec4<f32>;
	var py: vec4<f32>;
	var pz: vec4<f32>;
	var Sp: vec4<f32>;
	var Cp: vec4<f32>;
	var Ctp: vec4<f32>;
	var qx: vec4<f32>;
	var qy: vec4<f32>;
	var qz: vec4<f32>;
	var Sa: vec4<f32>;
	var Ca: vec4<f32>;

	if(alpha != 0.0)
	{
		px = Ct * sz_;
		py = St * sz_;
		pz = sz;
		Sp = sin(psi);
		Cp = cos(psi);
		Ctp = St*Sp - Ct*Cp;
		qx = mix( Ctp*St, Sp, sz);
		qy = mix(-Ctp*Ct, Cp, sz);
		qz = -(py*Cp + px*Sp);
		Sa = vec4<f32>(sin(alpha));
		Ca = vec4<f32>(cos(alpha));
		gx = Ca*px + Sa*qx;
		gy = Ca*py + Sa*qy;
		gz = Ca*pz + Sa*qz;
	}
	else
	{
		gx = Ct * sz_;
		gy = St * sz_;
		gz = sz;  
	}
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = dot(w3, gdotx);
	
	return 39.5 * n;
}

fn snoise3(x: vec3<f32>) -> f32
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
	
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;

	gx = Ct * sz_;
	gy = St * sz_;
	gz = sz;  
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = dot(w3, gdotx);
	
	return 39.5 * n;
}

fn snoise3offset(x: vec3<f32>, offset: f32) -> f32 {
    let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
    let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

    // Offset modifications
    let offset_x = offset * sin(x.x * 2.0 * 3.14159 / 100.0);
    let offset_y = offset * cos(x.y * 2.0 * 3.14159 / 100.0);
    let offset_z = offset * sin(x.z * 2.0 * 3.14159 / 100.0);
    let x_mod = vec3<f32>(x.x + offset_x, x.y + offset_y, x.z + offset_z);

    var uvw: vec3<f32> = M * x_mod;
    let i0 = floor(uvw);
    let f0 = uvw - i0;
    let gt_ = step(f0.xyx, f0.yzz);
    let lt_ = 1.0 - gt_;
    let gt = vec3<f32>(lt_.z, gt_.xy);
    let lt = vec3<f32>(lt_.xy, gt_.z);
    let o1 = min(gt, lt);
    let o2 = max(gt, lt);
    let i1 = i0 + o1;
    let i2 = i0 + o2;
    let i3 = i0 + vec3<f32>(1.0, 1.0, 1.0);

    let v0 = Mi * i0;
    let v1 = Mi * i1;
    let v2 = Mi * i2;
    let v3 = Mi * i3;

    let x0 = x_mod - v0;
    let x1 = x_mod - v1;
    let x2 = x_mod - v2;
    let x3 = x_mod - v3;

    // Continue with noise computation as before
    var hash: vec4<f32>;
    var theta: vec4<f32>;
    var sz: vec4<f32>;
    var psi: vec4<f32>;
    var St: vec4<f32>;
    var Ct: vec4<f32>;
    var sz_: vec4<f32>;

    hash = permute289v4f_psrn(permute289v4f_psrn(permute289v4f_psrn(
        vec4<f32>(i0.z, i1.z, i2.z, i3.z))
        + vec4<f32>(i0.y, i1.y, i2.y, i3.y))
        + vec4<f32>(i0.x, i1.x, i2.x, i3.x));
    theta = hash * 3.883222077;
    sz = hash * -0.006920415 + 0.996539792;
    psi = hash * 0.108705628;
    Ct = cos(theta);
    St = sin(theta);
    sz_ = sqrt(1.0 - sz*sz);

    var gx: vec4<f32>;
    var gy: vec4<f32>;
    var gz: vec4<f32>;
    gx = Ct * sz_;
    gy = St * sz_;
    gz = sz;  
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = dot(w3, gdotx);
	
	return 39.5 * n;
}

// Versions computing their gradient and returning a struct

fn psrdnoise3(x: vec3<f32>, p: vec3<f32>, alpha: f32) -> NG3
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
	
	var vx: vec4<f32>;
	var vy: vec4<f32>;
	var vz: vec4<f32>;

	if(any(p > vec3<f32>(0.0))) {
		vx = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
		vy = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
		vz = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
		if(p.x > 0.0) {
			vx = vx - floor(vx / p.x) * p.x;
		}
		if(p.y > 0.0) {
			vy = vy - floor(vy / p.y) * p.y;
		}
		if(p.z > 0.0) {
			vz = vz - floor(vz / p.z) * p.z;
		}
		i0 = floor(M * vec3<f32>(vx.x, vy.x, vz.x) + 0.5);
		i1 = floor(M * vec3<f32>(vx.y, vy.y, vz.y) + 0.5);
		i2 = floor(M * vec3<f32>(vx.z, vy.z, vz.z) + 0.5);
		i3 = floor(M * vec3<f32>(vx.w, vy.w, vz.w) + 0.5);
	}
	
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;
	var px: vec4<f32>;
	var py: vec4<f32>;
	var pz: vec4<f32>;
	var Sp: vec4<f32>;
	var Cp: vec4<f32>;
	var Ctp: vec4<f32>;
	var qx: vec4<f32>;
	var qy: vec4<f32>;
	var qz: vec4<f32>;
	var Sa: vec4<f32>;
	var Ca: vec4<f32>;

	if(alpha != 0.0)
	{
		px = Ct * sz_;
		py = St * sz_;
		pz = sz;
		Sp = sin(psi);
		Cp = cos(psi);
		Ctp = St*Sp - Ct*Cp;
		qx = mix( Ctp*St, Sp, sz);
		qy = mix(-Ctp*Ct, Cp, sz);
		qz = -(py*Cp + px*Sp);
		Sa = vec4<f32>(sin(alpha));
		Ca = vec4<f32>(cos(alpha));
		gx = Ca*px + Sa*qx;
		gy = Ca*py + Sa*qy;
		gz = Ca*pz + Sa*qz;
	}
	else
	{
		gx = Ct * sz_;
		gy = St * sz_;
		gz = sz;  
	}
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = 39.5 * dot(w3, gdotx);

	var dw: vec4<f32> = -6.0 * w2 * gdotx;
	var dn0: vec3<f32> = w3.x * g0 + dw.x * x0;
	var dn1: vec3<f32> = w3.y * g1 + dw.y * x1;
	var dn2: vec3<f32> = w3.z * g2 + dw.z * x2;
	var dn3: vec3<f32> = w3.w * g3 + dw.w * x3;
	var g: vec3<f32> = 39.5 * (dn0 + dn1 + dn2 + dn3);
	
	return NG3(n, g);
}

fn psdnoise3(x: vec3<f32>, p: vec3<f32>) -> NG3
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
	
	var vx: vec4<f32>;
	var vy: vec4<f32>;
	var vz: vec4<f32>;

	if(any(p > vec3<f32>(0.0))) {
		vx = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
		vy = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
		vz = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
		if(p.x > 0.0) {
			vx = vx - floor(vx / p.x) * p.x;
		}
		if(p.y > 0.0) {
			vy = vy - floor(vy / p.y) * p.y;
		}
		if(p.z > 0.0) {
			vz = vz - floor(vz / p.z) * p.z;
		}
		i0 = floor(M * vec3<f32>(vx.x, vy.x, vz.x) + 0.5);
		i1 = floor(M * vec3<f32>(vx.y, vy.y, vz.y) + 0.5);
		i2 = floor(M * vec3<f32>(vx.z, vy.z, vz.z) + 0.5);
		i3 = floor(M * vec3<f32>(vx.w, vy.w, vz.w) + 0.5);
	}
	
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;

	gx = Ct * sz_;
	gy = St * sz_;
	gz = sz;  
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = 39.5 * dot(w3, gdotx);

	var dw: vec4<f32> = -6.0 * w2 * gdotx;
	var dn0: vec3<f32> = w3.x * g0 + dw.x * x0;
	var dn1: vec3<f32> = w3.y * g1 + dw.y * x1;
	var dn2: vec3<f32> = w3.z * g2 + dw.z * x2;
	var dn3: vec3<f32> = w3.w * g3 + dw.w * x3;
	var g: vec3<f32> = 39.5 * (dn0 + dn1 + dn2 + dn3);
	
	return NG3(n, g);
}

fn srdnoise3(x: vec3<f32>, alpha: f32) -> NG3
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;
		
	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;
	var px: vec4<f32>;
	var py: vec4<f32>;
	var pz: vec4<f32>;
	var Sp: vec4<f32>;
	var Cp: vec4<f32>;
	var Ctp: vec4<f32>;
	var qx: vec4<f32>;
	var qy: vec4<f32>;
	var qz: vec4<f32>;
	var Sa: vec4<f32>;
	var Ca: vec4<f32>;

	if(alpha != 0.0)
	{
		px = Ct * sz_;
		py = St * sz_;
		pz = sz;
		Sp = sin(psi);
		Cp = cos(psi);
		Ctp = St*Sp - Ct*Cp;
		qx = mix( Ctp*St, Sp, sz);
		qy = mix(-Ctp*Ct, Cp, sz);
		qz = -(py*Cp + px*Sp);
		Sa = vec4<f32>(sin(alpha));
		Ca = vec4<f32>(cos(alpha));
		gx = Ca*px + Sa*qx;
		gy = Ca*py + Sa*qy;
		gz = Ca*pz + Sa*qz;
	}
	else
	{
		gx = Ct * sz_;
		gy = St * sz_;
		gz = sz;  
	}
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = 39.5 * dot(w3, gdotx);

	var dw: vec4<f32> = -6.0 * w2 * gdotx;
	var dn0: vec3<f32> = w3.x * g0 + dw.x * x0;
	var dn1: vec3<f32> = w3.y * g1 + dw.y * x1;
	var dn2: vec3<f32> = w3.z * g2 + dw.z * x2;
	var dn3: vec3<f32> = w3.w * g3 + dw.w * x3;
	var g: vec3<f32> = 39.5 * (dn0 + dn1 + dn2 + dn3);
	
	return NG3(n, g);
}

fn sdnoise3(x: vec3<f32>) -> NG3
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;

	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f_psrn( permute289v4f_psrn( permute289v4f_psrn( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;

	gx = Ct * sz_;
	gy = St * sz_;
	gz = sz;  
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = 39.5 * dot(w3, gdotx);

	var dw: vec4<f32> = -6.0 * w2 * gdotx;
	var dn0: vec3<f32> = w3.x * g0 + dw.x * x0;
	var dn1: vec3<f32> = w3.y * g1 + dw.y * x1;
	var dn2: vec3<f32> = w3.z * g2 + dw.z * x2;
	var dn3: vec3<f32> = w3.w * g3 + dw.w * x3;
	var g: vec3<f32> = 39.5 * (dn0 + dn1 + dn2 + dn3);
	
	return NG3(n, g);
}


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

const OCTAVES: i32 = 10; // {Octave count:0-11}
const CAVE_SIZE: f32 = 50.0; // {Entry room size:0-250}
const CAVE_THRESHOLD: f32 = 3.0;  // {How much to smooth the noise intersections:0-100}
const MAX_STEPS: i32  = 40; // {Max raymarch steps:0-300}
const NEAR_CLIP: f32 = 20.0; // {Near clipping distance: 0-1000}
const SURF_DIST: f32 = 0.005; // {Surface min distance steps:0-1}
const MAX_DIST: f32 = 300.0; // {Far clipping distance: 0-1000}

const PI: f32 = 3.141592653592;
const TAU: f32 = 6.283185307185;
const DUST_RADIUS: f32 = 0.1; // {Dust radii:0-1}
const NOISE_CHOICE: f32 = 3.0; // {Choise of noise algorithm:1-3}
const NOISE_OFFSET: f32 = 0.0001; // {Softening of noise: 0-1}
const MAX_DEPTH: f32 = 300.0; // {Dust radii:100-1500}
const SEA_LEVEL: f32 = 25.; // {The Z in the world where the sea surface is}
// allow dynamic C.OCTAVES

const MAX_OCTAVES: i32 = 5;
const MIN_OCTAVES: i32 = 1;
const OCTAVES_FALLOFF_DISTANCE: f32 = C.MAX_DIST;

// WGSL adaptation of K.jpg's Re-oriented 8-Point BCC Noise (OpenSimplex2S)
// Output: vec4<f32>(dF/dx, dF/dy, dF/dz, value)

// Permutes the input using a hash-like function
fn permute(t: vec4<f32>) -> vec4<f32> {
    return t * (t * 34.0 + 133.0);
}

// Computes a gradient based on a hash
fn grad(hash: f32) -> vec3<f32> {
    let cube: vec3<f32> = fract(vec3<f32>(hash / 1.0, hash / 2.0, hash / 4.0)) * 2.0 - 1.0;
    var cuboct: vec3<f32> = cube;
    cuboct[i32(hash / 16.0)] = 0.0;

    let type_: f32 = fract(hash / 8.0);
    let rhomb: vec3<f32> = (1.0 - type_) * cube + type_ * (cuboct + cross(cube, cuboct));
    var grad: vec3<f32> = cuboct * 1.22474487139 + rhomb;
    grad *= (1.0 - 0.042942436724648037 * type_) * 3.5946317686139184;

    return grad;
}

// Part of the BCC lattice noise computation
fn openSimplex2SDerivativesPart(X: vec3<f32>) -> vec4<f32> {
    let b: vec3<f32> = floor(X);
    let i4: vec4<f32> = vec4<f32>(X - b, 2.5);

    let v1: vec3<f32> = b + floor(dot(i4, vec4<f32>(0.25, 0.25, 0.25, 0.25)));
    let v2: vec3<f32> = b + vec3<f32>(1, 0, 0) + vec3<f32>(-1, 1, 1) * floor(dot(i4, vec4<f32>(-0.25, 0.25, 0.25, 0.35)));
    let v3: vec3<f32> = b + vec3<f32>(0, 1, 0) + vec3<f32>(1, -1, 1) * floor(dot(i4, vec4<f32>(0.25, -0.25, 0.25, 0.35)));
    let v4: vec3<f32> = b + vec3<f32>(0, 0, 1) + vec3<f32>(1, 1, -1) * floor(dot(i4, vec4<f32>(0.25, 0.25, -0.25, 0.35)));

    var hashes: vec4<f32> = permute(fract(vec4<f32>(v1.x, v2.x, v3.x, v4.x) / 289.0));
    hashes = permute(fract(hashes + vec4<f32>(v1.y, v2.y, v3.y, v4.y) / 289.0));
    hashes = fract(permute(fract(hashes + vec4<f32>(v1.z, v2.z, v3.z, v4.z) / 289.0)) / 48.0);

    let d1: vec3<f32> = X - v1; let d2: vec3<f32> = X - v2; let d3: vec3<f32> = X - v3; let d4: vec3<f32> = X - v4;
    let a: vec4<f32> = max(vec4<f32>(0.75 - dot(d1, d1), 0.75 - dot(d2, d2), 0.75 - dot(d3, d3), 0.75 - dot(d4, d4)), vec4<f32>(0.0, 0.0, 0.0, 0.0));

    let aa: vec4<f32> = a * a;
    let aaaa: vec4<f32> = aa * aa;

    let g1: vec3<f32> = grad(hashes.x); let g2: vec3<f32> = grad(hashes.y);
    let g3: vec3<f32> = grad(hashes.z); let g4: vec3<f32> = grad(hashes.w);

    let extrapolations: vec4<f32> = vec4<f32>(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
    // Manually calculate the weighted gradient matrix multiplication
let w = aa * a * extrapolations;
let derivative: vec3<f32> = vec3<f32>(
    -8.0 * (d1.x * w.x * g1.x + d2.x * w.y * g2.x + d3.x * w.z * g3.x + d4.x * w.w * g4.x),
    -8.0 * (d1.y * w.x * g1.y + d2.y * w.y * g2.y + d3.y * w.z * g3.y + d4.y * w.w * g4.y),
    -8.0 * (d1.z * w.x * g1.z + d2.z * w.y * g2.z + d3.z * w.z * g3.z + d4.z * w.w * g4.z)
) + vec3<f32>(
    g1.x * aaaa.x + g2.x * aaaa.y + g3.x * aaaa.z + g4.x * aaaa.w,
    g1.y * aaaa.x + g2.y * aaaa.y + g3.y * aaaa.z + g4.y * aaaa.w,
    g1.z * aaaa.x + g2.z * aaaa.y + g3.z * aaaa.z + g4.z * aaaa.w
);


    return vec4<f32>(derivative, dot(aaaa, extrapolations));
}

// Conventional derivatives computation
fn openSimplex2SDerivatives_Conventional(X: vec3<f32>) -> f32 {
    let modifiedX: vec3<f32> = dot(X, vec3<f32>(2.0 / 3.0)) - X;
    let result: vec4<f32> = openSimplex2SDerivativesPart(modifiedX) + openSimplex2SDerivativesPart(modifiedX + 144.5);

    return result.w;
}

// Improved XY derivatives computation
fn openSimplex2SDerivatives_ImproveXY(X: vec3<f32>) -> f32 {
    let orthonormalMap: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.788675134594813, -0.211324865405187, -0.577350269189626),
    vec3<f32>(-0.211324865405187, 0.788675134594813, -0.577350269189626),
    vec3<f32>(0.577350269189626, 0.577350269189626, 0.577350269189626)
);

    let modifiedX: vec3<f32> = orthonormalMap * X;
    let result: vec4<f32> = openSimplex2SDerivativesPart(modifiedX) + openSimplex2SDerivativesPart(modifiedX + 144.5);

    return result.w;
}


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

const N_SNOISE3D = 1;
const N_OPENSIMPLEX2S = 2;
const N_PSRDNOISE3 = 3;



fn vec3noise(choice: u32, p: vec3f, offset: f32) -> f32 {
    switch choice {
    case 1: {
        return snoise3D(p);
    }
    case 2: {
        return openSimplex2SDerivatives_ImproveXY(p);
    }
    case 3: {
        return snoise3offset(p, max(C.NOISE_OFFSET, offset));
    }
    default: {
    }
    }
    return 0.0;
}

fn customNoise3D(p: vec3f, amplitude: f32, scale: f32, octaves: i32, offset: f32) -> f32 {
    var value: f32 = 0.0;
    var currentAmplitude: f32 = amplitude;
    var currentScale: f32 = scale;

    for (var i: i32 = 0; i < octaves; i++) {
        value += currentAmplitude * vec3noise(u32(floor(C.NOISE_CHOICE)), p * currentScale, offset);
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


fn hashff(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn sdSphereNew(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

////////////////////////////////////////////////////////////////

fn mainScene(p: vec3f) -> f32 {
	 // Define the clipping plane at y = 0
    let planeNormal = vec3f(0.0, 1.0, 0.0);
    let planeHeight = 0.0; // Clipping exactly at y = 0

    // Calculate the distance to the clipping plane
    let planeDistance = sdPlane(p, planeNormal, planeHeight);

    // Early exit if the point is below the clipping plane
    // if (planeDistance < 0.0) {
    //     return planeDistance;
    // }
  let octaveCount = C.OCTAVES;
   /**
   EXPERIMENT: TWIST
//    */
//     let noiseValue = (snoise3(p * 3) + 1.0) / 2.0;
//  	let k: f32 = 0.01; // or some other amount
// 	let c: f32 = cos(k * p.y);
// 	let s: f32 = sin(k * p.y);
// 	var m: mat2x2<f32> = mat2x2<f32>(c, -s, s, c);
// 	var q: vec3<f32> = vec3<f32>(m * p.xz, p.y);


	// END EXPERIMENT
    let largerScaleNoise = customNoise3D(p, 6.0, 0.01, C.OCTAVES , C.NOISE_OFFSET);
    let solidVolumeSDF = sdSphere(p, vec3f(0.0), C.CAVE_SIZE);
    let caveSDF = opSmoothSubtraction(-largerScaleNoise, solidVolumeSDF, 50);
	// Define the clipping plane
    // Plane normal pointing up (0, 1, 0) for y = 0 plane
   
	
     // Return the result of the combined SDFs
    return opSmoothSubtraction(caveSDF, planeDistance, 10);
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

const numLights = 4;
const baseLightPower = 18.0;
const lights = array<vec3f, numLights>(
    vec3f(4.0, -2.0, -4.0),
    vec3f(-1, -.25, 1.),
    vec3f(0., -10.0, 0.),
    vec3f(0., 20.0, 0.)
);
const lightPowers = array<f32, numLights>( 4.0 , 1.0, 2.0, 1.0 );
const lightColors = array<vec3f, numLights>(
    vec3f(1.0, 0.9, 0.9),
    vec3f(1.0, 1.0, 1.0),
    vec3f(1.0, 1.0, 0.0),
    vec3f(1.0, 1.0, 0.0),
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
   
	let p = ro + rd * d;


	let skyGradient = mix(vec4f(0.0, 0.3, 0.8, 1.0), vec4f(0.6, 0.8, 1.0, 1.0),  p.y / 1000);
	let depthColor = select(mix(vec4f(0.0, 0.3, .8, 1.0), vec4f(0.0, 0.0, 0.0, 0.0), clamp(p.y / C.MAX_DEPTH, 0.0,1.0)),skyGradient, p.y < -C.SEA_LEVEL); // Pure blue at maximum distance
 	var fragColor = skyGradient;
    if (d <= C.MAX_DIST) {
        
        let N = getNormal(p);
        let V = -rd;

        // PBR shading parameters
        // Calculate noise generation direction based on the surface normal and a slight angle off of vertical
// Calculate noise generation direction based on the surface normal and a slight angle off of vertical
// Define noise coordinates based on the position p and a scaling factor
// Inside the if (d <= C.MAX_DIST) block
// Lighting calculations
        let ambientIntensity = 0.001;  // Increased ambient intensity
        let ambientColor = vec3f(0.2, 0.2, 0.2);  // Adjust color if needed
// Calculate noise generation direction based on the surface normal and a slight angle off of vertical
let noiseAngle = radians(15.0); // Adjust the angle as desired
let noiseDirection = normalize(vec3f(sin(noiseAngle), 0.0, cos(noiseAngle))); // Rotate around y-axis
let noiseCoord = vec3f(p.x, p.y, p.z) / 70.0; // Adjust the scaling factor as needed

// Adjust noise coordinates based on surface normal
let rotatedCoord = rotateVector(noiseCoord, N, noiseAngle) * 12;
let noiseValue = (snoise3(rotatedCoord + vec3f(0.0, 0.0, C.time * 2.0)) + 1.0) / 2.0; // Generate noise

// Compute the influence of the noise based on the dot product between surface normal and noise direction
let noiseInfluence = max(dot(N, -noiseDirection) * 5, 0.0);

// Modify the light contribution based on noise value and its influence
let causticMultiplier = 1.0 + noiseValue * noiseInfluence; // Adjust the multiplier as needed


        var albedo = mix(vec3f(1.0), vec3f(0.0, 0.1, 0.07), clamp(customNoise3D(p, 6.0, 0.1, C.OCTAVES, 0.0001 ) * 0.5,0.0, 0.15));
		albedo = mix(albedo, vec3f(0.0, 0.15, 0.05), clamp(customNoise3D(p, 6.0, 0.3, C.OCTAVES, C.NOISE_OFFSET) * 0.5,0.0, 0.3));
		albedo = mix(albedo, vec3f(0.0, 0.1, 0.03), clamp(customNoise3D(p, 6.0, 0.01, C.OCTAVES, C.NOISE_OFFSET) * 0.5,0.0, 0.8));
		albedo = mix(albedo, vec3f(0.0, 0.1, 0.03), clamp(p.y / C.MAX_DEPTH, 0.0,1.0));
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
            Lo += (vec3f(1.0) - F + specular) * radiance * max(dot(N, L), 0.0) * baseLightPower * lightPowers[i] * causticMultiplier;
        }

        let ambient = ambientColor * ambientIntensity;  // Apply ambient light
        var color = (ambient + Lo) * albedo;  // Combine lighting contributions
        color = color / (color + vec3f(1.0));  // Tone mapping
        color = pow(color, vec3f(1.0/2.2));  // Gamma correction
        fragColor = vec4(color, 1.0);

        // Modify the output color based on distance to make distant objects bluer
       
        let blueFactor = d / C.MAX_DIST; // Normalize distance to range [0, 1]
        fragColor = mix(fragColor, depthColor, blueFactor + min(causticMultiplier, 0.1)); // Blwwwwend original color with blue based on distance
		
    }

    return fragColor;
}

// Function to rotate a vector around an axis
fn rotateVector(v: vec3f, axis: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    let t = 1.0 - c;
    let x = v.x;
    let y = v.y;
    let z = v.z;
    let nx = axis.x;
    let ny = axis.y;
    let nz = axis.z;
    let m00 = t * nx * nx + c;
    let m01 = t * nx * ny - s * nz;
    let m02 = t * nx * nz + s * ny;
    let m10 = t * nx * ny + s * nz;
    let m11 = t * ny * ny + c;
    let m12 = t * ny * nz - s * nx;
    let m20 = t * nx * nz - s * ny;
    let m21 = t * ny * nz + s * nx;
    let m22 = t * nz * nz + c;
    return vec3f(
        m00 * x + m01 * y + m02 * z,
        m10 * x + m11 * y + m12 * z,
        m20 * x + m21 * y + m22 * z
    );
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
