use std::ptr::NonNull;
use std::ffi;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use super::{VkAllocator, MemoryBlock, MemoryType};

#[derive(Debug)]
pub enum BufferType {
    ComputeUpload
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    mem_block: MemoryBlock
}

impl Buffer {
    pub fn new(
        device: &DeviceLoader,
        vk_alloc: &mut VkAllocator,
        buf_type: BufferType,
        size: u64
    ) -> Result<Self> {
        let (vk_usage, mem_type) = match buf_type {
            BufferType::ComputeUpload => (
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryType::Cross
            )
        };
    
        // Create buffer
        let create_info = vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
    
        let buffer = unsafe { device.create_buffer(&create_info, None) }
            .result()
            .context("Failed to create buffer")?;
    
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
    
        // Allocate and bind memory
        let mem_block = vk_alloc.alloc(device, &mem_req, mem_type).context("Failed to allocate memory")?;
    
        unsafe { device.bind_buffer_memory(buffer, mem_block.mem, mem_block.region.offset) }
            .result()
            .context("Failed to bind memory")?;

        Ok(Self {
            buffer,
            mem_block
        })
    }

    pub fn ptr(&self) -> Option<NonNull<ffi::c_void>> {
        self.mem_block.ptr
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            vk_alloc.free(self.mem_block);
        }
    }
}