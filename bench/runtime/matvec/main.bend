import Base
# matvec: a dense matrix times a vector in f32, R times over -- the
# suite's flat Array<F32> case. Its inner loop is the dot product at the
# heart of a neural network's linear layer. The matrix A has N = 2^D rows
# of N numbers, row-major in one Array<F32>; the vectors x and y hold N
# numbers each.
#
# A run, with val(s, k) = unit(prng(prng((k + 1) * 2654435761 + s))) and
# unit(u) = ((u & 16777215) - 8388608) / 8388608, a number in [-1, 1)
# that f32 holds exactly:
# - fill: A is cut into 2^B blocks of rows and each block writes its own
#   entries; entry k of A is val(7, k), and entry j of x is val(11, j);
# - R rounds: each block computes its rows of y. Row i's dot product with
#   x, p, adds product j into partial sum j % 4, and the four are added
#   at the end as (s0 + s1) + (s2 + s3). A compiler may not reorder an
#   f32 sum, so the order is the program's: four sums keep four additions
#   in flight, where one running sum waits for each. Then y[i] =
#   val(bits(p), i), and x and y swap: a round's products are hashed into
#   the next round's vector, so every number stays in [-1, 1) for any R,
#   and one wrong bit in any product changes the checksum;
# - fold: the bits of the last vector, in order, into the checksum, h =
#   h * 2654435761 ^ bits.
# The arrays are sized once outside the `!` (Array.new fills on one
# lane). Knobs: D (from 2n up: a row is a multiple of four entries), B
# (the fork depth: 2^B blocks of rows, every B gives the same bits) and R.
#
# Unsafe: the lanes share A, x and y through Base's @unsafe Array.fork
# and Array.join, so --check-only says which defs rely on them. A round
# only reads A and x, and every lane writes its own rows of y.
#
# Sizes: small D = 6n, B = 3n, R = 4n (expect 2932539476, verified
# against the C twin), big D = 12n, B = 11n, R = 256n (expect
# 2172030964).

# Handles
# -------

# The matrix and the two vectors.
type Three is Type:
  Three{a: Array<F32>, x: Array<F32>, y: Array<F32>}

def three.join(l: Three, r: Three) -> Three:
  Three{la, lx, ly} = l
  Three{ra, rx, ry} = r
  Three{Array.join(F32, la, ra), Array.join(F32, lx, rx), Array.join(F32, ly, ry)}

def three.rejoin(a2: Array<F32>, x2: Array<F32>, y2: Array<F32>, r: Three) -> Three:
  Three{a, x, y} = r
  Three{Array.join(F32, a, a2), Array.join(F32, x, x2), Array.join(F32, y, y2)}

# x and y swapped: a round's result is the next round's vector.
def three.swap(r: Three) -> Three:
  Three{a, x, y} = r
  Three{a, y, x}

# Entries
# -------

# xorshift32 (13, 17, 5).
def prng(+x: U32) -> U32:
  +a = U32.xor(x, U32.shln(x, 13n))
  +b = U32.xor(a, U32.shrn(a, 17n))
  U32.xor(b, U32.shln(b, 5n))

# The low 24 bits of u as a number in [-1, 1).
def unit(u: U32) -> F32:
  F32.div(F32.sub(U32.to_f32(U32.and(u, 16777215)), 8388608.0), 8388608.0)

# Entry k of stream s.
def val(s: U32, k: U32) -> F32:
  unit(prng(prng(((k + 1) * 2654435761 + s : U32))))

# Entries k to k + n of stream s.
def fill.go(+n: Nat, +k: U32, +s: U32, v: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      Array.set(F32, v, k, val(s, k))
    case 1n+p:
      fill.go(p, (k + 1 : U32), s, Array.set(F32, v, k, val(s, k)))

# The fill's forks: d levels down to the blocks of 2^e entries under
# block i, with two handles to the array. Block i writes only its entries.
def fill(+d: Nat, +i: U32, +e: Nat, +s: U32, fv: Array<F32> & Array<F32>) -> Array<F32>:
  match d:
    case 0n:
      (v, v2) = fv
      Array.join(F32, fill.go(U32.to_nat((U32.shln(1, e) - 1 : U32)), U32.shln(i, e), s, v), v2)
    case 1n+p:
      (v0, v1) = fv
      +j = (i * 2 : U32)
      l r = fill(p, j, e, s, Array.fork(F32, v0)) fill(p, (j + 1 : U32), e, s, Array.fork(F32, v1))
      Array.join(F32, l, r)

# Rows
# ----

# A row's dot product, with the two arrays it read.
type Dot is Type:
  Dot{a: Array<F32>, x: Array<F32>, p: F32}

# r holds A and its entry v at ia, q holds x and its entry w at jx: the
# first sum takes v * w and the four rotate, then the n entries after
# them do the same. A row is a multiple of four entries, so its last
# product finds the sums rotated to (3, 0, 1, 2): the result is sums
# (0 + 1) + (2 + 3).
def dot.go(+n: Nat, +ia: U32, +jx: U32, s0: F32, s1: F32, s2: F32, s3: F32, r: Array<F32> & F32, q: Array<F32> & F32) -> Dot:
  match n:
    case 0n:
      (a, v) = r
      (x, w) = q
      Dot{a, x, F32.add(F32.add(s1, s2), F32.add(s3, F32.add(s0, F32.mul(v, w))))}
    case 1n+p:
      (a, v) = r
      (x, w) = q
      +ia1 = (ia + 1 : U32)
      +jx1 = (jx + 1 : U32)
      dot.go(p, ia1, jx1, s1, s2, s3, F32.add(s0, F32.mul(v, w)), Array.get(F32, a, ia1), Array.get(F32, x, jx1))

# Row i of A (2^lg entries from i * 2^lg) . x.
def dot(+i: U32, +lg: Nat, a: Array<F32>, x: Array<F32>) -> Dot:
  +ia = U32.shln(i, lg)
  dot.go(U32.to_nat((U32.shln(1, lg) - 1 : U32)), ia, 0, 0.0, 0.0, 0.0, 0.0, Array.get(F32, a, ia), Array.get(F32, x, 0))

# d holds A, x and row i's dot product p: y[i] takes val(bits(p), i).
def row.put(+i: U32, y: Array<F32>, d: Dot) -> Three:
  Dot{a, x, p} = d
  Three{a, x, Array.set(F32, y, i, val(F32.bits(p), i))}

# Row i of y.
def row(+i: U32, +lg: Nat, t: Three) -> Three:
  Three{a, x, y} = t
  row.put(i, y, dot(i, lg, a, x))

# The n rows of y from row i.
def rows(+n: Nat, +i: U32, +lg: Nat, t: Three) -> Three:
  match n:
    case 0n:
      t
    case 1n+p:
      rows(p, (i + 1 : U32), lg, row(i, lg, t))

# A round's forks: d levels down to the blocks of 2^e rows under block i.
# fa, fx and fy are two handles each to A, x and y. Block i reads A and x
# and writes only its rows of y, 2^e from i * 2^e: no two lanes write one
# slot, and none reads a slot another writes.
def mul(+d: Nat, +i: U32, +e: Nat, +lg: Nat, fa: Array<F32> & Array<F32>, fx: Array<F32> & Array<F32>, fy: Array<F32> & Array<F32>) -> Three:
  match d:
    case 0n:
      (a, a2) = fa
      (x, x2) = fx
      (y, y2) = fy
      three.rejoin(a2, x2, y2, rows(U32.to_nat(U32.shln(1, e)), U32.shln(i, e), lg, Three{a, x, y}))
    case 1n+p:
      (a0, a1) = fa
      (x0, x1) = fx
      (y0, y1) = fy
      +j = (i * 2 : U32)
      l r = mul(p, j, e, lg, Array.fork(F32, a0), Array.fork(F32, x0), Array.fork(F32, y0))
        mul(p, (j + 1 : U32), e, lg, Array.fork(F32, a1), Array.fork(F32, x1), Array.fork(F32, y1))
      three.join(l, r)

# Run
# ---

# n more rounds, at fork depth b over blocks of 2^e rows.
def reps(+n: Nat, +b: Nat, +e: Nat, +lg: Nat, t: Three) -> Three:
  match n:
    case 0n:
      t
    case 1n+p:
      Three{a, x, y} = t
      reps(p, b, e, lg, three.swap(mul(b, 0, e, lg, Array.fork(F32, a), Array.fork(F32, x), Array.fork(F32, y))))

# r holds a vector and its entry v at j: h folds v's bits, then those of
# the n entries after j.
def sum.go(+n: Nat, +j: U32, +h: U32, r: Array<F32> & F32) -> U32:
  match n:
    case 0n:
      (x, v) = r
      U32.xor((h * 2654435761 : U32), F32.bits(v))
    case 1n+p:
      (x, v) = r
      +q = (j + 1 : U32)
      sum.go(p, q, U32.xor((h * 2654435761 : U32), F32.bits(v)), Array.get(F32, x, q))

# The checksum of the last vector: x, after the last swap.
def fold(+lg: Nat, t: Three) -> U32:
  Three{a, x, y} = t
  sum.go(U32.to_nat((U32.shln(1, lg) - 1 : U32)), 0, 0, Array.get(F32, x, 0))

# A filled from stream 7 at fork depth b, x from stream 11, and y.
def start(+d: Nat, +b: Nat, a: Array<F32>, x: Array<F32>, y: Array<F32>) -> Three:
  Three{fill(b, 0, Nat.sub(Nat.add(d, d), b), 7, Array.fork(F32, a)), fill.go(U32.to_nat((U32.shln(1, d) - 1 : U32)), 0, 11, x), y}

# n rounds on a 2^d x 2^d matrix at fork depth b.
def run(+n: Nat, +d: Nat, +b: Nat, a: Array<F32>, x: Array<F32>, y: Array<F32>) -> U32:
  +fb = Nat.min(b, d)
  fold(d, reps(n, fb, Nat.sub(d, fb), d, start(d, fb, a, x, y)))

def size() -> Nat:
  12n

def depth() -> Nat:
  11n

def rounds() -> Nat:
  256n

# small: D = 6n, B = 3n, R = 4n expect 2932539476; big: D = 12n, B = 11n,
# R = 256n expect 2172030964
def main() -> IO(Unit):
  a = Array.new(F32, Nat.add(size(), size()), 0.0)
  x = Array.new(F32, size(), 0.0)
  y = Array.new(F32, size(), 0.0)
  IO.print(U32.show(run!(rounds(), size(), depth(), a, x, y)))
