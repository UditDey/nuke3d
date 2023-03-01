//! 2D vector graphics system for the renderer

use anyhow::{Result, Context};

use super::vk_core::VkCore;

/// A 2D point
#[repr(C)]
pub struct Point {
    pub x: f32,
    pub y: f32
}

/// 2D vector graphics canvas
pub struct Canvas2D {

}

impl Canvas2D {
    pub fn new(vk_core: &VkCore) -> Result<Self> {
        let queue_len = vk_core.frame_queue_len();

        

        Ok(Self {})
    }
}