use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use super::{PhysicalDeviceInfo, name_object, name_multiple};

pub fn create_command_buffers(
    device: &DeviceLoader,
    count: usize,
    phys_dev_info: &PhysicalDeviceInfo
) -> Result<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
    let create_info = vk::CommandPoolCreateInfoBuilder::new()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(phys_dev_info.gfx_queue_family());

    let cmd_pool = unsafe { device.create_command_pool(&create_info, None) }
        .result()
        .context("Failed to create command pool")?;

    name_object(device, cmd_pool.object_handle(), vk::ObjectType::COMMAND_POOL, "cmd_pool")?;

    let alloc_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count as u32);

    let cmd_bufs = unsafe { device.allocate_command_buffers(&alloc_info) }
        .result()
        .context("Failed to allocate command buffer")?
        .to_vec();

    name_multiple!(device, cmd_bufs.iter(), vk::ObjectType::COMMAND_BUFFER, "cmd_buf");

    Ok((cmd_pool, cmd_bufs))
}