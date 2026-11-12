// Native TypeScript twin of main.bend: a dense 2^D x 2^D matrix times a
// vector in f32 (Float32Array cells, Math.fround per op), R times over,
// single-threaded, 1-to-1 with the Bend program: the same entries
// (stream 7 for A, 11 for x), each round's row i . x in the same four
// partial sums, hashed into y[i] = val(bits, i), then x and y swap, and
// the same fold of the last vector's bits (imul + >>>0 for wrapping
// u32). B only shapes Bend's forks, so it has no twin here.
const D = 12;
const R = 256;
const f = Math.fround;
const word = new Uint32Array(1);
const real = new Float32Array(word.buffer);

function word_prng(x: number): number {
  const a = (x ^ (x << 13)) >>> 0;
  const b = (a ^ (a >>> 17)) >>> 0;
  return (b ^ (b << 5)) >>> 0;
}

// the low 24 bits of u as a number in [-1, 1)
function unit(u: number): number {
  return ((u & 16777215) - 8388608) / 8388608;
}

function val(s: number, k: number): number {
  return unit(word_prng(word_prng((Math.imul(k + 1, 2654435761) + s) >>> 0)));
}

// the bits of an f32
function bits(v: number): number {
  real[0] = v;
  return word[0];
}

// the row at `at` . x over n entries: product j goes to sum j % 4, and
// the four are added in pairs
function dot(n: number, a: Float32Array, at: number, x: Float32Array): number {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  for (let j = 0; j < n; j += 4) {
    s0 = f(s0 + f(a[at + j] * x[j]));
    s1 = f(s1 + f(a[at + j + 1] * x[j + 1]));
    s2 = f(s2 + f(a[at + j + 2] * x[j + 2]));
    s3 = f(s3 + f(a[at + j + 3] * x[j + 3]));
  }
  return f(f(s0 + s1) + f(s2 + s3));
}

// one round: every row of y from its dot product's bits
function mul(n: number, a: Float32Array, x: Float32Array,
  y: Float32Array): void {
  for (let i = 0; i < n; i++) {
    y[i] = val(bits(dot(n, a, i * n, x)), i);
  }
}

const n = 1 << D;
const a = new Float32Array(n * n);
let x = new Float32Array(n);
let y = new Float32Array(n);
let h = 0;
for (let k = 0; k < n * n; k++) {
  a[k] = val(7, k);
}
for (let j = 0; j < n; j++) {
  x[j] = val(11, j);
}
for (let r = 0; r < R; r++) {
  mul(n, a, x, y);
  const t = x;
  x = y;
  y = t;
}
for (let j = 0; j < n; j++) {
  h = (Math.imul(h, 2654435761) ^ bits(x[j])) >>> 0;
}
console.log(h);
