#![allow(clippy::many_single_char_names)]
use super::Backend;
use ndarray::*;
use ocl::builders::KernelBuilder;
use ocl::{Buffer, Program, Queue};
use std::fmt;

use paste::paste;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, GpuError>;

pub struct OpenCL {}

impl Backend for OpenCL {}

/// Various erros that can occur when working with an OpenCL backend
#[derive(Debug, Error)]
pub enum GpuError {
    #[error(transparent)]
    Ocl(WrappedOclError),
    #[error("ndarray memory layout not contiguous")]
    NotContiguousMemory,
    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
}

impl From<ocl::Error> for GpuError {
    fn from(err: ocl::Error) -> Self {
        GpuError::Ocl(WrappedOclError(err))
    }
}

#[derive(Debug)]
pub struct WrappedOclError(pub ocl::Error);

impl std::error::Error for WrappedOclError {}

impl fmt::Display for WrappedOclError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! impl_ops {
    (unary[$($(#[$unopmeta:meta])* $unop:ident, $unopname:literal, $unopcode:literal);*],
    binary[$($(#[$binopmeta:meta])* $binop:ident, $binopname:literal,
    $binopargs:literal,
    $binopcode:literal);*],
     ops[$($op:ident, $opname:literal,  $opcode:literal);*]
    ) => {
        pub static PROGRAM : &'static str = concat!(
            $(
             concat!("__kernel void ", $unopname, "(__global const float *src, __global float *res) {
                 int const idx = get_global_id(0);
                 float a = src[idx];
                 res[idx] =",
                $unopcode,
                ";\n}\n",
             ),
            )*
            $(
             concat!("__kernel void ", $binopname, "(__global const float *a_g, __global float *b_g,  __global float *res", $binopargs ,") {",
                 $binopcode,
                "\n}\n",
             ),
            )*

            $(
             concat!($opcode, "\n")
            ),*
        );

        /// Returns a [`ocl::ProQue`] with `PROGRAM` as source
        pub fn program() -> ocl::Result<ocl::ProQue>{
             ocl::ProQue::builder().src(PROGRAM).build()
        }

        $(
            paste!{
                 /// The name of the kernel function
                 pub const [<$op _KERNEL>]: &str = $opname;
            }
        )*

        /// Kernel Unary operations
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum UnaryOps {
            $(
                 $(#[$unopmeta])*
                $unop,
            )*
        }

        impl AsRef<str> for UnaryOps {
            fn as_ref(&self) -> &str {
                match self {
                    $(
                        UnaryOps::$unop => $unopname,
                    )*
                }
            }
        }

        impl From<UnaryOps> for String {
            fn from(ops: UnaryOps) -> String {
                ops.as_ref().to_string()
            }
        }

          $(

            paste! {
                $(#[$unopmeta])*
                pub fn [<$unopname ocl>]<D: Dimension>(
                    a: &Array<f32, D>,
                    prg: &Program,
                    queue: Queue,
                ) -> Result<Array<f32, D>> {
                    unary_op_ocl(UnaryOps::$unop, a, prg, queue)
                }
            }
        )*

        /// Kernel Binary operations
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum BinaryOps {
            $(
                $(#[$binopmeta])*
                $binop,
            )*
        }

        impl AsRef<str> for BinaryOps {
            fn as_ref(&self) -> &str {
                match self {
                    $(
                        BinaryOps::$binop => $binopname,
                    )*
                }
            }
        }

        impl From<BinaryOps> for String {
            fn from(ops: BinaryOps) -> String {
                ops.as_ref().to_string()
            }
        }

        $(
            paste! {
                $(#[$binopmeta])*
                pub fn [<$binopname ocl>]<D: Dimension>(
                    a: &Array<f32, D>,
                    b: &Array<f32, D>,
                    prg: &Program,
                    queue: Queue,
                ) -> Result<Array<f32, D>> {
                    binary_op_ocl(BinaryOps::$binop, a, b, prg, queue)
                }
            }
        )*
    };
}

// implements various operations as OpenCL kernel functions
//
// `unary` operation apply an instruction to each element in the `ndarray`
impl_ops!(
    unary[
         /// Returns a new array with same dimensions as `a` and
         /// `res[idx] = max(a[idx], 0)`
        ReLu, "relu_", "max(a, (float)0.)";
        /// Returns a new array with same dimensions as `a` and the natural logarithm
        /// applied to each value: `res[idx] = a[idx].ln()`
        Log, "ln_", "log(a)";
        /// Returns a new array with same dimensions as `a` and
        /// `res[idx] = exp(a[idx])` applied to each value
        Exp, "exp_", "exp(a)";
        /// Returns a new array with same dimensions as `a` but negated
        /// `res[idx] = -a[idx]`
        Neg, "neg_", "-a"
    ],
    binary[
        /// Returns a new matrix with `a + b` values
        Add, "add_", "", r#"
            int gid = get_global_id(0);
            res[gid] = a_g[gid] +  b_g[gid];
        "#;
          /// Returns a new matrix with `a - b` values
        Sub, "sub_", "", r#"
            int gid = get_global_id(0);
            res[gid] = a_g[gid] - b_g[gid];
        "#;
        /// Returns a new matrix with `pow(a,b)` values
        Pow, "pow_", "", r#"
            int gid = get_global_id(0);
            res[gid] = pow(a_g[gid], b_g[gid]);
        "#;
        /// Returns a new matrix with `pow(a,b) * log(a)` values
        PowXLogA, "pow_x_log_", "", r#"
            int gid = get_global_id(0);
            res[gid] = pow(a_g[gid], (float)b_g[gid]) * log(a_g[gid]);
        "#;
        /// Returns a new matrix with `b * (pow(a, (b-1.0))` values
        BxPow, "b_x_pow", "", r#"
            int gid = get_global_id(0);
            res[gid] = b_g[gid] * (pow((float)a_g[gid], (float)(b_g[gid]-1.0)));
        "#;
        /// Returns a new matrix filled with `1.0` where `a[idx] == b[idx]`
        /// and `0.` otherwise
        Eq, "eq_", "", r#"
            int gid = get_global_id(0);
            res[gid] = 1.0 * (a_g[gid] == b_g[gid]);
        "#;
        /// Returns a new matrix with `a/b` values
        Div, "div_", "", r#"
            int gid = get_global_id(0);
            res[gid] = a_g[gid] / b_g[gid];
        "#;
        /// Returns a new matrix with `a*b` values
        Mul, "mul_", "", r#"
            int gid = get_global_id(0);
            res[gid] = a_g[gid] * b_g[gid];
        "#;
        /// Returns a new matrix with `a * (b >= 0)` values
        ReLu, "relu_bin_", "", r#"
            int gid = get_global_id(0);
            res[gid] = a_g[gid] * (b_g[gid] >= 0);
        "#
        ],
    ops[MATMUL, "matmul", r#"
__kernel void matmul(
        int const Aheight, int const Awidth,
        int const Bheight, int const Bwidth,
        __global const float* A,
        __global const float* B,
        __global float* C )
        {
            int Y = get_global_id(0);
            int X = get_global_id(1);

            if( Y < Aheight && X < Bwidth ) {
                float ret = 0;
                for (int e = 0; e < Awidth; ++e)
                    ret += A[Y * Awidth + e] * B[e * Bwidth + X];

                C[Y * Bwidth + X] = ret;
            }
        }
    "#;
    MATMUL_SCALAR , "matmul_scalar", r#"
__kernel void matmul_scalar(
            __private float const w,
            __global float const* const src,
            __global float* const res)
{
    int const idx = get_global_id(0);
    res[idx] = src[idx] * w;
}
    "#
    ]
);

// https://github.com/csehydrogen/Winograd-OpenCL/blob/master/kernel.cl

/// Holds OpenCl objects required when executing kernel functions
#[derive(Debug, Clone)]
pub struct OpenCLKernelCtx {
    /// the opencl program that contains the UnaryOps kernel code
    pub prg: Program,
    /// The queue used for enqueuing a command
    pub queue: Queue,
}

impl OpenCLKernelCtx {
    pub fn split(self) -> (Program, Queue) {
        (self.prg, self.queue)
    }
    /// Execute a known unary kernel function
    pub fn unary_op<D: Dimension>(self, a: &Array<f32, D>, ops: UnaryOps) -> Result<Array<f32, D>> {
        let (prg, queue) = self.split();
        unary_op_ocl(ops, a, &prg, queue)
    }
}

pub fn binary_op_ocl<D: Dimension>(
    ops: BinaryOps,
    a: &Array<f32, D>,
    b: &Array<f32, D>,
    prg: &Program,
    queue: Queue,
) -> Result<Array<f32, D>> {
    if a.ndim() == b.ndim() && a.shape() == b.shape() {
        let len = a.shape().iter().product();
        // copy the input into the opencl buffer
        let a_buf = host_slice_buf(a, queue.clone(), len)?;
        let b_buf = host_slice_buf(b, queue.clone(), len)?;

        let res_buf = Buffer::<f32>::builder()
            .queue(queue.clone())
            .len(len)
            .build()?;

        let kernel = KernelBuilder::new()
            .name(ops)
            .program(prg)
            .queue(queue)
            .global_work_size(len)
            .arg(&a_buf)
            .arg(&b_buf)
            .arg(&res_buf)
            .build()?;

        unsafe {
            kernel.enq()?;
        }
        let mut res = vec![0.; len];
        res_buf.read(&mut res).enq()?;

        Ok(Array::from_shape_vec(a.raw_dim(), res)?)
    } else {
        Err(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into())
    }
}

/// Executes a known unary operation with OpenCL on every element in the ndarray
/// `a`
fn unary_op_ocl<D: Dimension>(
    ops: UnaryOps,
    a: &Array<f32, D>,
    prg: &Program,
    queue: Queue,
) -> Result<Array<f32, D>> {
    // that's the total number of elements
    let len = a.shape().iter().product();
    // copy the input into the opencl buffer
    let src_buf = host_slice_buf(a, queue.clone(), len)?;

    let res_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(len)
        .build()?;

    let kernel = KernelBuilder::new()
        .name(ops)
        .program(prg)
        .queue(queue)
        .global_work_size(len)
        .arg(&src_buf)
        .arg(&res_buf)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    // the output mat has (m , n) dimension
    let mut res = vec![0.; len];
    res_buf.read(&mut res).enq()?;

    Ok(Array::from_shape_vec(a.raw_dim(), res)?)
}

/// matmul of two 2D matrices `A * W`
pub fn matmul_2x2(
    a: &Array<f32, Ix2>,
    w: &Array<f32, Ix2>,
    prg: &Program,
    queue: Queue,
) -> Result<Array2<f32>> {
    matmul_ocl(a, w, a.dim(), w.dim(), prg, queue)
}

/// Perform matmul of matrix `a` and column vector `w`
pub fn matmul_2x1(
    a: &Array<f32, Ix2>,
    w: &Array<f32, Ix1>,
    prg: &Program,
    queue: Queue,
) -> Result<Array1<f32>> {
    let res = matmul_ocl_vec(a, w, a.dim(), (w.dim(), 1), prg, queue)?;
    Ok(Array1::from_vec(res))
}

/// perform matmul of row vector `a` and matrix `w`
pub fn matmul_1x2(
    a: &Array<f32, Ix1>,
    w: &Array<f32, Ix2>,
    prg: &Program,
    queue: Queue,
) -> Result<Array1<f32>> {
    let res = matmul_ocl_vec(a, w, (1, a.dim()), w.dim(), prg, queue)?;
    Ok(Array1::from_vec(res))
}

/// perform dot product of 1D arrays.
///
/// The dot product is a sum of the elementwise products.
pub fn matmul_1x1(
    a: &Array<f32, Ix1>,
    w: &Array<f32, Ix1>,
    prg: &Program,
    queue: Queue,
) -> Result<f32> {
    let res = matmul_ocl_vec(a, w, (1, a.dim()), (w.dim(), 1), prg, queue)?;
    Ok(res.into_iter().sum())
}

/// multiply A*W
fn matmul_ocl<A, W>(
    a: &Array<f32, A>,
    w: &Array<f32, W>,
    a_shape: (usize, usize),
    w_shape: (usize, usize),
    prg: &Program,
    queue: Queue,
) -> Result<Array2<f32>>
where
    A: Dimension,
    W: Dimension,
{
    let ((m, _), (_, n)) = (a_shape, w_shape);
    let res = matmul_ocl_vec(a, w, a_shape, w_shape, prg, queue)?;
    // the shape of the matmul
    Ok(Array2::from_shape_vec((m, n), res)?)
}

/// multiply A*W into a vec
/// TODO: this is limited to 2D, should behave:
///  https://pytorch.org/docs/stable/generated/torch.matmul.html
fn matmul_ocl_vec<A, W>(
    a: &Array<f32, A>,
    w: &Array<f32, W>,
    a_shape: (usize, usize),
    w_shape: (usize, usize),
    prg: &Program,
    queue: Queue,
) -> Result<Vec<f32>>
where
    A: Dimension,
    W: Dimension,
{
    // height / width
    let ((m, k), (k2, n)) = (a_shape, w_shape);
    if k != k2 {
        panic!(
            "inputs {} × {} and {} × {} are not compatible for matrix multiplication",
            m, k, k2, n
        )
    }
    let c_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(m * n)
        .build()?;

    // input buffer that use `a` and `w` memory directly
    let a_buf = host_slice_buf(a, queue.clone(), m * k)?;
    let w_buf = host_slice_buf(w, queue.clone(), k2 * n)?;

    let kernel = KernelBuilder::new()
        .name(MATMUL_KERNEL)
        .program(prg)
        .queue(queue)
        .global_work_size((m, n))
        .arg(&m)
        .arg(&k)
        .arg(&k2)
        .arg(&n)
        .arg(&a_buf)
        .arg(&w_buf)
        .arg(&c_buf)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    // the output mat has (m , n) dimension
    let mut out = vec![0.; m * n];
    c_buf.read(&mut out).enq()?;
    Ok(out)
}

/// Matrix * scalar : A * w
pub fn matmul_scalar<D: Dimension>(
    a: &Array<f32, D>,
    w: f32,
    prg: &Program,
    queue: Queue,
) -> Result<Array<f32, D>> {
    // that's the total number of elements
    let len = a.shape().iter().product();
    // copy the input into the opencl buffer
    let src_buf = host_slice_buf(a, queue.clone(), len)?;

    let res_buf = Buffer::<f32>::builder()
        .queue(queue.clone())
        .len(len)
        .build()?;

    let kernel = KernelBuilder::new()
        .name(MATMUL_SCALAR_KERNEL)
        .program(prg)
        .queue(queue)
        .global_work_size(len)
        .arg(&w)
        .arg(&src_buf)
        .arg(&res_buf)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut res = vec![0.; len];
    res_buf.read(&mut res).enq()?;

    Ok(Array::from_shape_vec(a.raw_dim(), res)?)
}

/// Creates a new OpenCL buffer that borrows array `a`'s memory
///
/// ### Safety
///
/// See [`Buffer::use_host_slice`], a must outlive the create buffer
fn host_slice_buf<D: Dimension>(
    a: &Array<f32, D>,
    queue: Queue,
    len: usize,
) -> Result<Buffer<f32>> {
    let mut buf = Buffer::<f32>::builder().queue(queue).len(len);

    buf = unsafe {
        buf.use_host_slice(
            a.as_slice_memory_order()
                .ok_or(GpuError::NotContiguousMemory)?,
        )
    };
    Ok(buf.build()?)
}

#[cfg(test)]
mod tests {
    use ndarray_rand::{
        rand::{self, Rng},
        rand_distr::Uniform,
        RandomExt,
    };

    use super::*;

    fn fmt_with_precision(a: f32, b: f32, precision: usize) -> (String, String) {
        (
            format!("{:.1$}", a, precision),
            format!("{:.1$}", b, precision),
        )
    }

    // compare the values with limited precision
    fn assert_eq_arrays<D>(a: &Array<f32, D>, b: &Array<f32, D>)
    where
        D: Dimension,
    {
        Zip::from(a).and(b).for_each(|&a, &b| {
            let mut precision = 5;
            let mut equal = false;
            while precision > 0 {
                let (sa, sb) = fmt_with_precision(a, b, precision);
                equal = sa == sb;
                if equal {
                    break;
                }
                precision -= 1;
            }
            if !equal {
                let (sa, sb) = fmt_with_precision(a, b, precision);
                assert_eq!(sa, sb);
            }
        });
    }

    #[test]
    fn matmul_2x2_opencl_test() {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let k: usize = rand::thread_rng().gen_range(1..100);
            let m: usize = rand::thread_rng().gen_range(1..100);
            let n: usize = rand::thread_rng().gen_range(1..100);

            let a = Array2::<f32>::random((m, k), Uniform::new(0., 10.));
            let b = Array2::<f32>::random((k, n), Uniform::new(0., 10.));
            let axb = a.dot(&b);

            let res = matmul_2x2(&a, &b, prg, queue.clone()).unwrap();
            assert_eq_arrays(&axb, &res);
        }
    }

    #[test]
    fn matmul_2x1_opencl_test() {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let k: usize = rand::thread_rng().gen_range(1..100);
            let m: usize = rand::thread_rng().gen_range(1..100);

            let a = Array2::<f32>::random((m, k), Uniform::new(0., 10.));
            let b = Array1::<f32>::random(k, Uniform::new(0., 10.));
            let axb = a.dot(&b);

            let res = matmul_2x1(&a, &b, prg, queue.clone()).unwrap();
            assert_eq_arrays(&axb, &res);
        }
    }

    #[test]
    fn matmul_1x2_opencl_test() {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let k: usize = rand::thread_rng().gen_range(1..100);
            let n: usize = rand::thread_rng().gen_range(1..100);

            let a = Array1::<f32>::random(k, Uniform::new(0., 10.));
            let b = Array2::<f32>::random((k, n), Uniform::new(0., 10.));
            let axb = a.dot(&b);

            let res = matmul_1x2(&a, &b, prg, queue.clone()).unwrap();
            assert_eq_arrays(&axb, &res);
        }
    }

    #[test]
    fn matmul_1x1_opencl_test() {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let k: usize = rand::thread_rng().gen_range(1..100);
            let a = Array1::<f32>::random(k, Uniform::new(0., 10.));
            let b = Array1::<f32>::random(k, Uniform::new(0., 10.));
            let axb = a.dot(&b);
            let res = matmul_1x1(&a, &b, prg, queue.clone()).unwrap();

            assert!((axb.abs() - res.abs().abs()) < 1.);
        }
    }

    #[test]
    fn matmul_scalar_opencl_test() {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let l: usize = rand::thread_rng().gen_range(1..10);
            let m: usize = rand::thread_rng().gen_range(1..10);
            let k: usize = rand::thread_rng().gen_range(1..100);

            let n: f32 = rand::thread_rng().gen_range(0.0..10000.);
            let a = Array1::<f32>::random(k, Uniform::new(0., 1337.));
            let nd = a.clone() * n;
            let res = matmul_scalar(&a, n, prg, queue.clone()).unwrap();
            assert_eq_arrays(&nd, &res);

            let a = Array2::<f32>::random((l, m), Uniform::new(0., 1337.));
            let nd = a.clone() * n;
            let res = matmul_scalar(&a, n, prg, queue.clone()).unwrap();
            assert_eq_arrays(&nd, &res);

            let a = Array3::<f32>::random((l, m, k), Uniform::new(0., 1337.));
            let nd = a.clone() * n;
            let res = matmul_scalar(&a, n, prg, queue.clone()).unwrap();
            assert_eq_arrays(&nd, &res);
        }
    }

    fn unary_ops_test<F>(ops: UnaryOps, f: F)
    where
        F: Fn(f32) -> f32,
    {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let m: usize = rand::thread_rng().gen_range(1..100);
            let n: usize = rand::thread_rng().gen_range(1..100);
            let a = Array2::<f32>::random((m, n), Uniform::new(0., 10.));
            let mut nd = a.clone();
            nd.iter_mut().for_each(|a| *a = f(*a));
            let ocl = unary_op_ocl(ops, &a, prg, queue.clone()).unwrap();
            assert_eq_arrays(&nd, &ocl);
        }
    }

    fn binary_ops_test<F>(ops: BinaryOps, expected: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let m: usize = rand::thread_rng().gen_range(1..133);
            let n: usize = rand::thread_rng().gen_range(1..420);
            let a = Array2::<f32>::random((m, n), Uniform::new(0., 10.));
            let b = Array2::<f32>::random((m, n), Uniform::new(0., 10.));

            let mut nd = a.clone();

            Zip::from(&mut nd).and(&b).for_each(|x, y| {
                *x = expected(*x, *y);
            });

            let ocl = binary_op_ocl(ops, &a, &b, prg, queue.clone()).unwrap();
            assert_eq_arrays(&nd, &ocl);
        }
    }

    fn binary_ops_test_relaxed<F>(ops: BinaryOps, f: F, threshold: f32)
    where
        F: Fn(f32, f32) -> f32,
    {
        let pro_que = program().unwrap();
        let prg = pro_que.program();
        let queue = pro_que.queue();
        for _ in 0..100 {
            let m: usize = rand::thread_rng().gen_range(1..133);
            let n: usize = rand::thread_rng().gen_range(1..420);
            let a = Array2::<f32>::random((m, n), Uniform::new(0., 5.));
            let b = Array2::<f32>::random((m, n), Uniform::new(0., 5.));

            let mut nd = a.clone();

            Zip::from(&mut nd).and(&b).for_each(|x, y| {
                *x = f(*x, *y);
            });

            let ocl = binary_op_ocl(ops, &a, &b, prg, queue.clone()).unwrap();
            Zip::from(&nd).and(&ocl).for_each(|&nd, &ocl| {
                assert!((nd.abs() - ocl.abs().abs()) < threshold);
            });
        }
    }

    #[test]
    fn log_opencl_test() {
        unary_ops_test(UnaryOps::Log, |a| a.ln());
    }

    #[test]
    fn exp_opencl_test() {
        unary_ops_test(UnaryOps::Exp, |a| a.exp());
    }

    #[test]
    fn neg_opencl_test() {
        unary_ops_test(UnaryOps::Neg, |a| -a);
    }

    #[test]
    fn relu_opencl_test() {
        unary_ops_test(UnaryOps::ReLu, |a| if a > 0. { a } else { 0. });
    }

    #[test]
    fn binary_add_opencl_test() {
        binary_ops_test(BinaryOps::Add, |a, b| a + b);
    }

    #[test]
    fn binary_eq_opencl_test() {
        binary_ops_test(BinaryOps::Eq, |a, b| {
            if (a - b).abs() < 0.0000001 {
                1.0
            } else {
                0.0
            }
        });
    }

    #[test]
    fn binary_mul_opencl_test() {
        binary_ops_test(BinaryOps::Mul, |a, b| a * b);
    }

    #[test]
    fn binary_div_opencl_test() {
        binary_ops_test(BinaryOps::Div, |a, b| a / b);
    }

    #[test]
    fn binary_sub_opencl_test() {
        binary_ops_test(BinaryOps::Sub, |a, b| a - b);
    }

    #[test]
    fn binary_relu_opencl_test() {
        binary_ops_test(BinaryOps::ReLu, |a, b| if b >= 0. { a } else { 0. });
    }

    // has some float precision deviations (different float math)
    #[test]
    fn binary_pow_opencl_test() {
        binary_ops_test_relaxed(BinaryOps::Pow, |a, b| a.powf(b), 5.);
    }

    // has some float precision deviations (different float math)
    #[test]
    fn binary_pow_log_opencl_test() {
        binary_ops_test_relaxed(BinaryOps::PowXLogA, |a, b| a.powf(b) * a.ln(), 5.);
    }

    // has some float precision deviations (different float math)
    #[test]
    fn binary_b_xpow_log_opencl_test() {
        binary_ops_test_relaxed(BinaryOps::BxPow, |a, b| b * a.powf(b - 1.), 5.);
    }
}
