use crate::ops::gpu::OpenCLKernelCtx;

use ndarray::{Array, Array2, Dimension};

// Requirements:
//  - Tensor without backend
//  - Tensor with backend context, "executable tensor"
//  - context holds necessary past data and kernel stuff in case of opencl
//  - graph object that stores the computational graph
//  - each tensor represents a node in the graph, when tensor requires_grad then
//    .grad is another tensor holding the gradiant of x with respect to some
//    scalar value

// closure like ag::with( )

pub type TensorData<D> = Array<f32, D>;

pub struct Tensor<D> {
    /// Data of the tensor
    data: TensorData<D>,
    /// The device used for execution
    device: Device,
    /// Whether every operation on the tensor should be tracked.
    requires_grad: bool,

    /// Record of the tensors with all executed operations
    saved_tensors: Vec<Tensor<D>>,

    /// The gradient of the tensor's data w.r.t itself
    grad: Option<TensorData<D>>,
}

impl<D: Dimension> Tensor<D> {
    // TODO add feature that always enables require_grad

    pub fn zeros() {}
    pub fn ones() {}
    pub fn randn() {}
    pub fn arange() {}
    pub fn uniform<Dim: Dimension>(_shape: Dim) {}

    pub fn dot<W: Dimension>(&self, _w: &Array<f32, W>) -> Array2<f32> {
        todo!()
    }

    pub fn sqrt(&self) {}

    pub fn assign(&mut self) {
        todo!("replace inner data")
    }

    pub fn div(&self) {}

    pub fn sigmoid(&self) {}

    /// Calculates the gradients and stores them in the respective tensors
    pub fn backward(&mut self) {}
}

impl<D> From<Tensor<D>> for TensorData<D> {
    fn from(t: Tensor<D>) -> TensorData<D> {
        t.data
    }
}

impl<D> AsRef<TensorData<D>> for Tensor<D> {
    fn as_ref(&self) -> &TensorData<D> {
        &self.data
    }
}

pub trait TensorExt<D> {
    fn cpu(self) -> Tensor<D>;

    fn opencl(self, kernel: OpenCLKernelCtx) -> Tensor<D>;
}

/// Represents a device which can execute Tensor operations
#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    OpenCL(OpenCLKernelCtx),
}

impl From<OpenCLKernelCtx> for Device {
    fn from(ctx: OpenCLKernelCtx) -> Self {
        Device::OpenCL(ctx)
    }
}
