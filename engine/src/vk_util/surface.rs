use std::ffi::c_void;

use erupt::{vk, InstanceLoader};
use anyhow::{Result, Context};

#[cfg(unix)]
use xcb::Xid;

use crate::platform::WindowInfo;

pub fn create_surface(instance: &InstanceLoader, window_info: &WindowInfo) -> Result<vk::SurfaceKHR> {
    #[cfg(unix)]
    let create_info = vk::XcbSurfaceCreateInfoKHRBuilder::new()
        .connection(window_info.conn.get_raw_conn() as *mut c_void)
        .window(window_info.window.resource_id());

    unsafe {
        #[cfg(unix)]
        instance.create_xcb_surface_khr(&create_info, None)
    }
    .result()
    .context("Failed to create surface")
}