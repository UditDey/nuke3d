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
    canvas_2d: Canvas2D,
    vk_core: VkCore
}

impl Renderer {
    pub fn new(config: &RendererConfig, window: &dyn Window) -> Result<Self> {
        let vk_core = VkCore::setup(config, window)?;
        let canvas_2d = Canvas2D::new(&vk_core)?;

        Ok(Self {
            canvas_2d,
            vk_core
        })
    }

    pub fn destroy(self) {
        self.canvas_2d.destroy(&self.vk_core);
        self.vk_core.destroy();
    }
}