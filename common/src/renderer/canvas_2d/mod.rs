//! GPU accelerated 2D vector graphics canvas

mod renderer;
mod recorder;

pub use renderer::Canvas2DRenderer;
pub use recorder::{Canvas2DRecorder, InitState, ContourState};