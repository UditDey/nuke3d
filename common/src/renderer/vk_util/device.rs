use ash::{vk, extensions::khr, Instance, Device};
use anyhow::{Result, Context};

use super::phys_dev::{DEVICE_EXTS, PhysicalDeviceInfo};

/// Device extensions functions
pub struct DeviceExts {
    swapchain_ext: khr::Swapchain,
    maintenance_1_ext: khr::Maintenance1
}

impl DeviceExts {
    /// `VK_KHR_swapchain` extension functions
    pub fn swapchain_ext(&self) -> &khr::Swapchain {
        &self.swapchain_ext
    }

    /// `VK_KHR_maintenance1` extension functions
    pub fn maintenance_1_ext(&self) -> &khr::Maintenance1 {
        &self.maintenance_1_ext
    }
}

/// Create a logical device, load its extension functions and get the graphics queue
///
/// Since [`Device`] is a huge struct, its boxed to avoid using too much stack space
///
/// Enabled device features:
/// - sampler anisotropy
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

    let dev_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&DEVICE_EXTS)
        .enabled_features(&dev_features);

    let device = unsafe { instance.create_device(phys_dev, &create_info, None) }
        .context("Failed to create device")?;

    let gfx_queue = unsafe { device.get_device_queue(phys_dev_info.gfx_queue_family(), 0) };

    let device_exts = DeviceExts {
        swapchain_ext: khr::Swapchain::new(instance, &device),
        maintenance_1_ext: khr::Maintenance1::new(instance, &device)
    };

    Ok((Box::new(device), device_exts, gfx_queue))
}