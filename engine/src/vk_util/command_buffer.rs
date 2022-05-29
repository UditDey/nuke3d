use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use super::PhysicalDeviceInfo;

pub fn create_command_buffers(
    device: &DeviceLoader,
    count: usize,
    phys_dev_info: &PhysicalDeviceInfo
) -> Result<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
    let create_info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(phys_dev_info.gfx_queue_family);

    let cmd_pool = unsafe { device.create_command_pool(&create_info, None) }
        .result()
        .context("Failed to create command pool")?;

    let alloc_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count as u32);

    let cmd_bufs = unsafe { device.allocate_command_buffers(&alloc_info) }
        .result()
        .context("Failed to allocate command buffer")?;

    Ok((cmd_pool, cmd_bufs.to_vec()))
}