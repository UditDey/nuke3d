use erupt::{vk, InstanceLoader, DeviceLoader};
use anyhow::{Result, Context};

use super::{PhysicalDeviceInfo, DEVICE_EXTS};

pub fn create_device(
    instance: &InstanceLoader,
    phys_dev: vk::PhysicalDevice,
    phys_dev_info: &PhysicalDeviceInfo
) -> Result<(DeviceLoader, vk::Queue)> {
    let queue_create_infos = [
        vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(phys_dev_info.gfx_queue_family)
            .queue_priorities(&[1.0])
    ];

    let dev_features = vk::PhysicalDeviceFeaturesBuilder::new().sampler_anisotropy(true);

    let dev_create_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&DEVICE_EXTS)
        .enabled_features(&dev_features);

    let device = unsafe { DeviceLoader::new(&instance, phys_dev, &dev_create_info) }
        .context("Failed to create device")?;

    let gfx_queue = unsafe { device.get_device_queue(phys_dev_info.gfx_queue_family, 0) };

    Ok((device, gfx_queue))
}