use std::ffi::CStr;
use std::os::raw::c_char;

use erupt::{vk, InstanceLoader};
use anyhow::{Result, Context};

pub const DEVICE_EXTS: [*const c_char; 2] = [
    vk::KHR_SWAPCHAIN_EXTENSION_NAME,

    #[allow(deprecated)]
    vk::KHR_MAINTENANCE1_EXTENSION_NAME
];

pub struct PhysicalDeviceInfo {
    pub gfx_queue_family: u32,
    pub props: vk::PhysicalDeviceProperties,
    pub mem_props: vk::PhysicalDeviceMemoryProperties
}

pub fn pick_physical_device(
    instance: &InstanceLoader,
    surface: vk::SurfaceKHR
) -> Result<(vk::PhysicalDevice, PhysicalDeviceInfo)> {
    // Shortlist eligible physical devices
    struct EligibleDeviceInfo {
        phys_dev: vk::PhysicalDevice,
        queue_family: u32,
        props: vk::PhysicalDeviceProperties
    }

    let mut elig_devs = vec![];

    let phys_devs = unsafe { instance.enumerate_physical_devices(None) }
        .result()
        .context("Failed to get available physical devices")?;

    'outer:
    for &phys_dev in &phys_devs {
        // Must have a graphics queue compatible with the surface
        let queue_props = unsafe {
            instance.get_physical_device_queue_family_properties(phys_dev, None)
        };

        let mut queue_family_opt = None;

        for (i, prop) in queue_props.iter().enumerate() {
            let cond_1 = prop.queue_flags.contains(vk::QueueFlags::GRAPHICS);

            let cond_2 = unsafe {
                instance
                    .get_physical_device_surface_support_khr(phys_dev, i as u32, surface)
                    .result()
                    .context("Failed to get physical device surface support")?
            };

            if cond_1 && cond_2 {
                queue_family_opt = Some(i as u32);
                break;
            }
        }

        let queue_family = match queue_family_opt {
            Some(family) => family,
            None => continue 'outer
        };

        // Must supprt required extensions
        let avail_exts = unsafe {
            instance
                .enumerate_device_extension_properties(phys_dev, None, None)
                .result()
                .context("Failed to get physical device extension properties")?
        };

        for req_ext in DEVICE_EXTS {
            let req_ext_str = unsafe { CStr::from_ptr(req_ext) };
            let mut found = false;

            for avail_ext in &avail_exts {
                let avail_ext_str = unsafe { CStr::from_ptr(avail_ext.extension_name.as_ptr()) };

                if avail_ext_str == req_ext_str {
                    found = true;
                    break;
                }
            }

            if !found {
                continue 'outer;
            }
        }

        let props = unsafe { instance.get_physical_device_properties(phys_dev) };

        elig_devs.push(EligibleDeviceInfo {
            phys_dev,
            queue_family,
            props
        });
    }

    // Pick a physical device, preferring discrete GPUs
    let picked_dev = elig_devs
        .iter()
        .max_by_key(|elig_dev| match elig_dev.props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 2,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 0
        })
        .context("Failed to find suitable physical device")?;

    let mem_props = unsafe { instance.get_physical_device_memory_properties(picked_dev.phys_dev) };

    let info = PhysicalDeviceInfo {
        gfx_queue_family: picked_dev.queue_family,
        props: picked_dev.props,
        mem_props
    };

    Ok((picked_dev.phys_dev, info))
}