pub trait Optimizer {
    /// Initiates the gradient descent.
    ///
    /// Adjusts each parameter by its gradient.
    fn step(&mut self);
}
