use std::ffi;
use std::ptr::NonNull;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context, bail};

use super::{VkAllocator, MemoryBlock, MemoryType};

#[derive(PartialEq)]
pub enum BufferType {
    ComputeStorage,
    Staging
}

impl BufferType {
    fn usage(&self) -> vk::BufferUsageFlags {
        match self {
            Self::ComputeStorage => vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            Self::Staging => vk::BufferUsageFlags::TRANSFER_SRC
        }
    }

    fn mem_type(&self) -> MemoryType {
        match self {
            Self::ComputeStorage => MemoryType::Device,
            Self::Staging => MemoryType::Host
        }
    }
}

pub struct Buffer {
    buf: vk::Buffer,
    block: MemoryBlock
}

impl Buffer {
    pub fn new(
        device: &DeviceLoader,
        vk_alloc: &mut VkAllocator,
        buf_type: BufferType,
        size: u64
    ) -> Result<Self> {
        // Create buffer
        let create_info = vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(buf_type.usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buf = unsafe { device.create_buffer(&create_info, None) }
            .result()
            .context("Failed to create buffer")?;

        let mem_req = unsafe { device.get_buffer_memory_requirements(buf) };

        // Allocate and bind memory
        let block = vk_alloc.alloc(device, &mem_req, buf_type.mem_type())
            .context("Failed to allocate memory")?;

        unsafe { device.bind_buffer_memory(buf, block.mem(), block.offset()) }
            .result()
            .context("Failed to bind buffer memory")?;

        Ok(Self { buf, block })
    }

    pub fn buf(&self) -> vk::Buffer {
        self.buf
    }

    pub fn ptr(&self) -> Result<*mut ffi::c_void> {
        self.block
            .ptr()
            .map(NonNull::as_ptr)
            .context("This buffer does not have a mapped pointer")
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        unsafe { device.destroy_buffer(self.buf, None); }
        vk_alloc.free(self.block);
    }
}

pub struct UploadBuffer {
    target_buf: Buffer,
    stg_buf: Buffer,
    size: u64
}

impl UploadBuffer {
    pub fn new(
        device: &DeviceLoader,
        vk_alloc: &mut VkAllocator,
        target_buf_type: BufferType,
        size: u64
    ) -> Result<Self> {
        // The target buf_type can't be Staging, that wouldn't make sense
        if target_buf_type == BufferType::Staging {
            bail!("UploadBuffer can't be made with BufferType::Staging")
        }
        
        // Create target and staging buffers
        let target_buf = Buffer::new(device, vk_alloc, target_buf_type, size)
            .context("Failed to create target buffer")?;

        let stg_buf = Buffer::new(device, vk_alloc, BufferType::Staging, size)
            .context("Failed to create staging buffer")?;

        Ok(Self {
            target_buf,
            stg_buf,
            size
        })
    }

    pub fn target_buf(&self) -> vk::Buffer {
        self.target_buf.buf()
    }

    pub fn ptr(&self) -> *mut ffi::c_void {
        self.stg_buf.ptr().unwrap() // Should never fail
    }

    pub fn cmd_upload(&self, device: &DeviceLoader, cmd_buf: vk::CommandBuffer) {
        let region = vk::BufferCopyBuilder::new()
            .src_offset(0)
            .dst_offset(0)
            .size(self.size);

        unsafe { device.cmd_copy_buffer(cmd_buf, self.stg_buf.buf(), self.target_buf.buf(), &[region]); }
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        self.target_buf.destroy(device, vk_alloc);
        self.stg_buf.destroy(device, vk_alloc);
    }
}