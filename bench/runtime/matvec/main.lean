-- Native Lean twin of main.bend: a dense 2^D x 2^D matrix times a vector
-- in Float32, R times over, single-threaded, 1-to-1 with the Bend
-- program: the same entries (stream 7 for A, 11 for x), each round's row
-- i . x in the same four partial sums, hashed into y[i] = val(bits, i),
-- then x and y swap, and the same fold of the last vector's bits. A, x
-- and y are FloatArrays (Lean's unboxed float array: an f64 cell holds
-- an f32 exactly), updated in place while uniquely referenced. B only
-- shapes Bend's forks, so it has no twin here.
def D : Nat := 12
def R : Nat := 256

def word_prng (x : UInt32) : UInt32 :=
  let a := x ^^^ (x <<< (13 : UInt32))
  let b := a ^^^ (a >>> (17 : UInt32))
  b ^^^ (b <<< (5 : UInt32))

-- the low 24 bits of u as a number in [-1, 1)
def unit (u : UInt32) : Float32 :=
  ((u &&& 16777215).toNat.toFloat32 - 8388608.0) / 8388608.0

def val (s : UInt32) (k : Nat) : Float32 :=
  unit (word_prng (word_prng ((UInt32.ofNat k + 1) * 2654435761 + s)))

-- the row at `row` . x from entry j on: product j goes to sum j % 4, and
-- the four are added in pairs (tail recursion keeps Float32 unboxed)
partial def dot (n : Nat) (a x : FloatArray) (row j : Nat)
    (s0 s1 s2 s3 : Float32) : Float32 :=
  if j < n then
    dot n a x row (j + 4)
      (s0 + a[row + j]!.toFloat32 * x[j]!.toFloat32)
      (s1 + a[row + j + 1]!.toFloat32 * x[j + 1]!.toFloat32)
      (s2 + a[row + j + 2]!.toFloat32 * x[j + 2]!.toFloat32)
      (s3 + a[row + j + 3]!.toFloat32 * x[j + 3]!.toFloat32)
  else
    (s0 + s1) + (s2 + s3)

-- one round: every row of y from its dot product's bits
def mul (n : Nat) (a x y0 : FloatArray) : FloatArray := Id.run do
  let mut y := y0
  for i in [0:n] do
    let p := dot n a x (i * n) 0 0.0 0.0 0.0 0.0
    y := y.set! i (val p.toBits i).toFloat
  return y

def main : IO Unit := do
  let n := 2 ^ D
  let mut a : FloatArray := FloatArray.emptyWithCapacity (n * n)
  let mut x : FloatArray := FloatArray.emptyWithCapacity n
  let mut y : FloatArray := FloatArray.emptyWithCapacity n
  let mut h : UInt32 := 0
  for k in [0:n * n] do
    a := a.push (val 7 k).toFloat
  for j in [0:n] do
    x := x.push (val 11 j).toFloat
    y := y.push 0.0
  for _ in [0:R] do
    let t := mul n a x y
    y := x
    x := t
  for j in [0:n] do
    h := (h * 2654435761) ^^^ x[j]!.toFloat32.toBits
  IO.println h
