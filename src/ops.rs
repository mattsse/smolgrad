use crate::tensor::Tensor;
pub use cpu::Cpu;
pub use gpu::OpenCL;
use std::marker::PhantomData;

/// Support for mathematical operations
pub trait Backend {}

pub enum Device {
    Cpu,
    Gpu,
    Other(Box<dyn Backend>),
}

pub mod cpu;
pub mod gpu;

// 1. User constructs tensors -> tensor symbols
// 2. constructs the forward graph by executing binary ops and those symbols
// 3. graph.forward(forward graph) does the actual computation
//    3.1 execute the graph by calling `forward` on every node in the graph
//    3.2 `forward` should return the output tensors and the backward type

// TODO: split forward and backward into two traits where forward returns Box<dyn Backward>

/// The computational graph
///
/// Creates and tracks the internal tensors
pub struct Graph {
    /// device used to execute the operations
    device: Device,
    /// Each tensor represents a node in the graph
    nodes: Vec<()>,

    tensors: Arena<Tensor<IxDynImpl>>,

    storage: Arena<Box<dyn Any>>,

    funs: Vec<FunctionContext<()>>,
}

#[derive(Clone)]
pub struct TensorRef {
    /// The internal index of the tensor arena.
    idx: Index,
    /// optional symbol to contextualise the tensor.
    symbol: Option<Cow<'static, str>>,
}

/// The part of the autograd operator function that implements the forward pass which operate on the tensors
pub trait Forward {
    /// Computes output tensors from input tensors.
    ///
    /// Forward pass is called with the input data tensor and returns the tensor
    /// containing the output.
    /// The `Context` is used to cache data that is then available during
    /// `backpass`.
    fn forward(self, graph: &mut Graph);
}

/// The part of the autograd operator function that implements the backward pass
pub trait Backward {
    fn backward(&mut self, graph: &mut Graph);
}

#[derive(Debug, Copy, Clone)]
pub struct AddForward;

pub enum Term {
    Tensor(TensorRef),
    Scalar(f32),
    Term(Box<Term>),
}

// registry https://github.com/iqlusioninc/abscissa/blob/e49ef7b61786e15f223c5f9dcdd5890003835d2e/core/src/component/registry.rs

// https://github.com/iqlusioninc/abscissa/pull/424/files

pub trait Func {
    /// Receives the gradient of the output tensor with respect to some scalar
    /// value and computes the gradient of the input tensors with respect to the
    /// same scalar value.
    ///
    /// Backward pass called with the tensor containing the gradient of the loss
    /// with respect to the output.
    /// `backward` computes the gradient of the loss with respect to the input.
    // input_grads: GradientVec, output_grad: GradientD
    fn backward(&self, graph: &mut Graph);
}

// Tensors require away to call into the graph
// The graph consists of a list functions, a tensor
// A tensor merely keeps a ref to the tensordata that is stored in the graph?

/// An instance of an autograd operator that holds the graphs tensors
pub struct FunctionContext<D> {
    marker: PhantomData<D>,

    parents: Vec<()>,

    /// Record of the tensors with all executed operations
    saved_tensors: Vec<Tensor<D>>,
}

impl<D> FunctionContext<D> {
    fn do_apply(lhs: &Tensor<D>, rhs: &Tensor<D>) {}

    fn apply(&mut self) {}

    /// Cache additional items that are then available during the backward pass.
    fn save_for_backward(&mut self) {}

    /// The input tensor
    fn input(&self) -> &Tensor<D> {
        todo!()
    }
}

/// An autograd operator function that implements the forward and backward
/// passes which operate on the tensors
pub trait Function {
    // TODO type should allow different ways to be applied.
    fn apply() {}
    /// Computes output tensors from input tensors.
    ///
    /// Forward pass is called with the input data tensor and returns the tensor
    /// containing the output.
    /// The `Context` is used to cache data that is then available during
    /// `backpass`.
    fn forward<D>(&mut self, ctx: &mut FunctionContext<D>);

    /// Receives the gradient of the output tensor with respect to some scalar
    /// value and computes the gradient of the input tensors with respect to the
    /// same scalar value.
    ///
    /// Backward pass called with the tensor containing the gradient of the loss
    /// with respect to the output.
    /// `backward` computes the gradient of the loss with respect to the input.
    // input_grads: GradientVec, output_grad: GradientD
    fn backward<D>(&mut self, ctx: &mut FunctionContext<D>);
}

/// Supported math operations
pub enum Operation {
    Add,
    Conv2D,
    Exp,
    Function,
    Log,
    Matmul,
    Max,
    Mul,
    Pow,
    ReLU,
    Reshape,
    Slice,
    Sub,
    Sum,
    Transpose,
}

use generational_arena::{Arena, Index};
use ndarray::IxDynImpl;
use std::any::Any;
use std::borrow::Cow;
use std::fmt;

/// Identifier for an individual component
///
/// This should ideally match the Rust path name to the corresponding type.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct Id(&'static str);

impl Id {
    /// Create a new component identifier
    pub const fn new(id: &'static str) -> Id {
        Id(id)
    }
}

impl AsRef<str> for Id {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
