use ash::{vk, extensions::khr, Instance, Device};
use anyhow::{Result, Context};

use super::phys_dev::{DEVICE_EXTS, PhysicalDeviceInfo};

/// Device extensions functions
pub struct DeviceExts {
    swapchain_ext: khr::Swapchain
}

impl DeviceExts {
    /// `VK_KHR_swapchain` extension functions
    pub fn swapchain_ext(&self) -> &khr::Swapchain {
        &self.swapchain_ext
    }
}

/// Create a logical device, load its extension functions and get the graphics queue
///
/// Since [`Device`] is a huge struct, its boxed to avoid using too much stack space
///
/// Enabled 1.0 device features:
/// - sampler anisotropy
/// - shader int16
/// - shader int8
///
/// Enabled VK_KHR_16bit_storage features
/// - storage buffer 16 bit access
/// - uniform and storage buffer 16 bit access
pub fn create_device(
    instance: &Instance,
    phys_dev: vk::PhysicalDevice,
    phys_dev_info: &PhysicalDeviceInfo
) -> Result<(Box<Device>, DeviceExts, vk::Queue)> {
    let queue_create_infos = [
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(phys_dev_info.gfx_queue_family())
            .queue_priorities(&[1.0])
            .build()
    ];
    
    let dev_features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .shader_int16(true);
        
    let mut dev_16_bit_storage_features = vk::PhysicalDevice16BitStorageFeatures::builder()
        .storage_buffer16_bit_access(true)
        .uniform_and_storage_buffer16_bit_access(true);

    let create_info = vk::DeviceCreateInfo::builder()
        .push_next(&mut dev_16_bit_storage_features)
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&DEVICE_EXTS)
        .enabled_features(&dev_features);

    let device = unsafe { instance.create_device(phys_dev, &create_info, None) }
        .context("Failed to create device")?;

    let gfx_queue = unsafe { device.get_device_queue(phys_dev_info.gfx_queue_family(), 0) };

    let device_exts = DeviceExts {
        swapchain_ext: khr::Swapchain::new(instance, &device)
    };

    Ok((Box::new(device), device_exts, gfx_queue))
}