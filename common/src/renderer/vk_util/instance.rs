use std::ffi::{CString, CStr};

use ash::{vk, extensions::khr, Entry, Instance};
use anyhow::{bail, Result, Context};

use crate::window::{Window, SurfaceCreateInfo};

const VK_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);

/// The platform specific surface extension functions
pub enum PlatformSurfaceExt {
    /// `VK_KHR_xlib_surface` extension functions
    Xlib(khr::XlibSurface)
}

/// Instance extension functions
pub struct InstanceExts {
    surface_ext: khr::Surface,
    platform_surface_ext: PlatformSurfaceExt
}

impl InstanceExts {
    /// `VK_KHR_surface` extension functions
    pub fn surface_ext(&self) -> &khr::Surface {
        &self.surface_ext
    }

    /// Platform specific surface extension functions
    pub fn platform_surface_ext(&self) -> &PlatformSurfaceExt {
        &self.platform_surface_ext
    }
}

/// Create the vulkan instance and load needed extension functions
pub fn create_instance(entry: &Entry, window: &dyn Window, force_validation: bool) -> Result<(Instance, InstanceExts)> {
    // Required instance extensions
    let mut req_exts = vec![khr::Surface::name().as_ptr()];

    match window.surface_create_info() {
        SurfaceCreateInfo::Xlib(_) => req_exts.push(khr::XlibSurface::name().as_ptr())
    }

    // Get available instance extensions
    let avail_exts = entry
        .enumerate_instance_extension_properties(None)
        .context("Failed to get available instance extensions")?;

    // Ensure required extensions are available
    for &req_ext in &req_exts {
        let req_ext_name = unsafe { CStr::from_ptr(req_ext) };

        let found = avail_exts
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
            .any(|ext_name| ext_name == req_ext_name);

        if !found {
            bail!("Required instance extension {req_ext_name:?} not available");
        }
    }

    // Required instance layers
    let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();

    let mut req_layers = vec![];

    if cfg!(debug_assertions) || force_validation {
        req_layers.push(validation_layer_name.as_ptr());
    }

    // Get available instance layers
    let avail_layers = entry
        .enumerate_instance_layer_properties()
        .context("Failed to get available instance layers")?;

    // Ensure required layers are available
    for &req_layer in &req_layers {
        let req_layer_name = unsafe { CStr::from_ptr(req_layer) };

        let found = avail_layers
            .iter()
            .map(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) })
            .any(|layer_name| layer_name == req_layer_name);

        if !found {
            bail!("Required instance layer {req_layer_name:?} not available");
        }
    }

    // Create instance
    let engine_name = CString::new("Nuke3D").unwrap();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(&engine_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(VK_VERSION);

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&req_exts)
        .enabled_layer_names(&req_layers);

    let instance = unsafe { entry.create_instance(&create_info, None) }
        .context("Failed to create instance")?;

    // Load instance extensions
    let instance_exts = InstanceExts {
        surface_ext: khr::Surface::new(entry, &instance),

        platform_surface_ext: match window.surface_create_info() {
            SurfaceCreateInfo::Xlib(_) => PlatformSurfaceExt::Xlib(khr::XlibSurface::new(entry, &instance))
        }
    };

    Ok((instance, instance_exts))
}