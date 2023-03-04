use std::ffi;

use ash::{vk, Device};
use anyhow::{bail, Result, Context};

use super::{VkCore, vma::{AllocInfo, VmaBuffer}};

/// Tool for transferring data from host to device
pub struct TransferBuffer {
    staging_buf: Option<VmaBuffer>,
    dest_buf: VmaBuffer,
    ptr: *mut ffi::c_void,
    size: u64
}

impl TransferBuffer {
    /// Note: `create_info` must have [`vk::BufferUsageFlags::TRANSFER_DST`] flag set
    pub fn new(vk_core: &VkCore, create_info: &vk::BufferCreateInfo) -> Result<Self> {
        if !create_info.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
            bail!("TRANSFER_DST flag not set");
        }

        // Create destination buffer
        let alloc_info = AllocInfo::new()
            .prefer_device()
            .mapped()
            .sequential_access()
            .allow_transfer_instead();

        let dest_buf = vk_core
            .vma_alloc()
            .create_buffer(create_info, &alloc_info)
            .context("Failed to create target buffer")?;

        // Check if destination buffer is mapped; if not, create staging buffer
        let staging_buf = if dest_buf.ptr().is_none() {
            let alloc_info = AllocInfo::new()
                .prefer_host()
                .mapped();

            let staging_create_info = vk::BufferCreateInfo::builder()
                .size(create_info.size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let staging_buf = vk_core
                .vma_alloc()
                .create_buffer(&staging_create_info, &alloc_info)
                .context("Failed to create staging buffer")?;

            Some(staging_buf)
        }
        else {
            None
        };

        let ptr = match &staging_buf {
            Some(staging_buf) => {
                staging_buf
                    .ptr()
                    .context("Staging buffer somehow not mapped")?
                    .as_ptr()
            },
            None => dest_buf.ptr().unwrap().as_ptr() // If staging_buf is None then this has to be Some
        };

        Ok(Self {
            staging_buf,
            dest_buf,
            ptr,
            size: create_info.size
        })
    }

    /// The size of the buffer
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The destination (device local) buffer
    pub fn buf(&self) -> vk::Buffer {
        self.dest_buf.buf()
    }

    /// Host mapped pointer to transfer data
    ///
    /// Writes to this should be sequential write-only since this may be in uncached memory
    pub fn ptr(&self) -> *mut ffi::c_void {
        self.ptr
    }

    /// Record this in a command buffer to ensure the data is transferred
    pub fn cmd_transfer(&self, device: &Device, cmd_buf: vk::CommandBuffer) {
        if let Some(staging_buf) = &self.staging_buf {
            let region = vk::BufferCopy::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(self.size)
                .build();

            unsafe { device.cmd_copy_buffer(cmd_buf, staging_buf.buf(), self.dest_buf.buf(), &[region]) };
        }
    }

    pub fn destroy(self, vk_core: &VkCore) {
        vk_core.vma_alloc().destroy_buffer(self.dest_buf);

        if let Some(staging_buf) = self.staging_buf {
            vk_core.vma_alloc().destroy_buffer(staging_buf);
        }
    }
}