//! Vulkan functionality

mod instance;
mod surface;
mod phys_dev;
mod device;
mod frame_queue;
mod cmd_buf;
mod vma;
mod buffer;

use ash::{vk, Entry, Instance, Device};
use anyhow::{Result, Context};

use crate::window::Window;
use super::RendererConfig;

use instance::{create_instance, InstanceExts};
use surface::create_surface;
use phys_dev::pick_physical_device;
use device::{create_device, DeviceExts};
use frame_queue::FrameQueue;
use cmd_buf::create_command_buffers;
use vma::VmaAllocator;

pub use buffer::TransferBuffer;

/// Container for the core vulkan objects
pub struct VkCore {
    vma_alloc: VmaAllocator,
    cmd_bufs: Vec<vk::CommandBuffer>,
    cmd_pool: vk::CommandPool,
    frame_queue: FrameQueue,
    gfx_queue: vk::Queue,
    device_exts: DeviceExts,
    device: Box<Device>,
    surface: vk::SurfaceKHR,
    instance_exts: InstanceExts,
    instance: Instance,
    _entry: Entry
}

impl VkCore {
    /// Create the core vulkan objects
    pub fn setup(config: &RendererConfig, window: &dyn Window) -> Result<Self> {
        // Load vulkan
        let entry = unsafe { Entry::load().context("Failed to load vulkan")? };

        // Create vulkan objects
        let (instance, instance_exts) = create_instance(&entry, window, config.force_validation)?;
        let surface = create_surface(&instance_exts, window)?;
        let (phys_dev, phys_dev_info) = pick_physical_device(&instance, &instance_exts, surface, &config.device_name)?;

        println!("Using device: {}", phys_dev_info.device_name());

        let (device, device_exts, gfx_queue) = create_device(&instance, phys_dev, &phys_dev_info)?;
        let frame_queue = FrameQueue::new(window, &instance_exts, surface, phys_dev, &device, &device_exts)?;

        println!("Frame queue length: {}", frame_queue.len());

        let (cmd_pool, cmd_bufs) = create_command_buffers(&device, &phys_dev_info, frame_queue.len())?;

        let vma_alloc = VmaAllocator::new(&instance, phys_dev, &device)?;

        Ok(Self {
            vma_alloc,
            cmd_bufs,
            cmd_pool,
            frame_queue,
            gfx_queue,
            device_exts,
            device,
            surface,
            instance_exts,
            instance,
            _entry: entry
        })
    }

    /// The vulkan [`Device`]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The vulkan memory allocator
    pub fn vma_alloc(&self) -> &VmaAllocator {
        &self.vma_alloc
    }

    /// The [`FrameQueue`]
    pub fn frame_queue(&self) -> &FrameQueue {
        &self.frame_queue
    }

    pub fn destroy(self) {
        unsafe {
            self.vma_alloc.destroy();
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.frame_queue.destroy(&self.device, &self.device_exts);
            self.device.destroy_device(None);
            self.instance_exts.surface_ext().destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}