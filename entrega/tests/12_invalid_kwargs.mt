@triton.jit
def kwarg(x): {
  y = tl.load(x, mask=1);
}
