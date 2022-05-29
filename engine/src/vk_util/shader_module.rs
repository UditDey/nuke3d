use std::ptr;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

pub fn create_shader_module(device: &DeviceLoader, spirv: &[u8]) -> Result<vk::ShaderModule> {
    // The struct takes a *const u32 but we have a [u8] so the builder won't work here
    let create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: spirv.len(),
        p_code: spirv.as_ptr() as *const u32
    };

    unsafe {
        device.create_shader_module(&create_info, None)
            .result()
            .context("Failed to create shader module")
    }
}