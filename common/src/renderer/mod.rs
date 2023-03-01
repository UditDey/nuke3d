//! Renderer for the engine and editor

mod vk_core;
mod canvas_2d;

use anyhow::Result;

use crate::window::Window;
use vk_core::VkCore;
use canvas_2d::Canvas2D;

/// Renderer configuration
pub struct RendererConfig<'a> {
    pub device_name: Option<&'a str>,
    pub force_validation: bool
}

/// The nuke3d renderer
pub struct Renderer {
    _vk_core: VkCore
}

impl Renderer {
    pub fn new(config: &RendererConfig, window: &dyn Window) -> Result<Self> {
        let vk_core = VkCore::setup(config, window)?;
        let canvas_2d = Canvas2D::new(&vk_core)?;

        Ok(Self { _vk_core: vk_core })
    }
}