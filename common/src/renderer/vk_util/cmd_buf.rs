use ash::{vk, Device};
use anyhow::{Result, Context};

use super::phys_dev::PhysicalDeviceInfo;

/// Create a command pool and the specified number of command buffers
pub fn create_command_buffers(
    device: &Device,
    phys_dev_info: &PhysicalDeviceInfo,
    count: u32
) -> Result<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
    // Create command pool
    let create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(phys_dev_info.gfx_queue_family());

    let cmd_pool = unsafe { device.create_command_pool(&create_info, None) }
        .context("Failed to create command pool")?;

    // Allocate command buffers
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);

    let cmd_bufs = unsafe { device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?;

    Ok((cmd_pool, cmd_bufs))
}