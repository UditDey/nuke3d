use ash::vk;
use anyhow::{Result, Context};

use crate::window::{Window, SurfaceCreateInfo};
use super::instance::{InstanceExts, PlatformSurfaceExt};

/// Create the vulkan surface for the window
pub fn create_surface(instance_exts: &InstanceExts, window: &dyn Window) -> Result<vk::SurfaceKHR> {
    match window.surface_create_info() {
        SurfaceCreateInfo::Xlib(create_info) => unsafe {
            let PlatformSurfaceExt::Xlib(xlib_ext) = instance_exts.platform_surface_ext();

            xlib_ext.create_xlib_surface(create_info, None).context("Failed to create surface")
        }
    }
}