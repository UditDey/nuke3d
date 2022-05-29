use std::ffi::{CString, CStr};

use erupt::{vk, InstanceLoader, EntryLoader};
use anyhow::{Result, Context, bail};

pub fn create_instance(entry: &EntryLoader) -> Result<InstanceLoader> {
    // Required instance extensions
    let req_exts = [
        vk::KHR_SURFACE_EXTENSION_NAME,
        
        #[cfg(unix)]
        vk::KHR_XCB_SURFACE_EXTENSION_NAME,

        #[cfg(debug_assertions)]
        vk::EXT_DEBUG_UTILS_EXTENSION_NAME
    ];

    let avail_exts = unsafe { entry.enumerate_instance_extension_properties(None, None) }
        .result()
        .context("Failed to get available instance extensions")?;

    for &req_ext in &req_exts {
        let req_ext_str = unsafe { CStr::from_ptr(req_ext) };

        let found = avail_exts
            .iter()
            .any(|avail_ext| unsafe {
                CStr::from_ptr(avail_ext.extension_name.as_ptr()) == req_ext_str
            });

        if !found {
            bail!("Required instance extension {req_ext_str:?} not available");
        }
    }

    // Required instance layers
    let validation_layer_name = CString::new("VK_LAYER_KHRONOS_validation").unwrap();

    let req_layers = [
        #[cfg(debug_assertions)]
        validation_layer_name.as_ptr()
    ];

    let avail_layers = unsafe { entry.enumerate_instance_layer_properties(None) }
        .result()
        .context("Failed to get available instance layers")?;

    for &req_layer in &req_layers {
        let req_layer_str = unsafe { CStr::from_ptr(req_layer) };

        let found = avail_layers
            .iter()
            .any(|avail_layer| unsafe {
                CStr::from_ptr(avail_layer.layer_name.as_ptr()) == req_layer_str
            });

        if !found {
            bail!("Required instance layer {req_layer_str:?} not available");
        }
    }

    // Create instance
    let app_engine_name = CString::new("Nuke3D").unwrap();

    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&app_engine_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&app_engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let create_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&req_exts)
        .enabled_layer_names(&req_layers);

    unsafe { InstanceLoader::new(entry, &create_info) }.context("Failed to create instance")
}