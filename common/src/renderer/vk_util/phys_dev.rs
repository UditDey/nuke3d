use std::borrow::Cow;
use std::ffi::{self, CStr};

use ash::{vk, extensions::khr, Instance};
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use anyhow::{bail, Result, Context};

use super::instance::InstanceExts;

/// Required device extensions
pub const DEVICE_EXTS: [*const ffi::c_char; 2] = [
    khr::Swapchain::name().as_ptr(),
    khr::Maintenance1::name().as_ptr()
];

/// Information associated with a physical device
pub struct PhysicalDeviceInfo {
    gfx_queue_family: u32,
    props: vk::PhysicalDeviceProperties,
    mem_props: vk::PhysicalDeviceMemoryProperties
}

impl PhysicalDeviceInfo {
    /// Queue family index supporting graphics queues.
    pub fn gfx_queue_family(&self) -> u32 {
        self.gfx_queue_family
    }

    /// The [`vk::PhysicalDeviceProperties`] of the physical device.
    pub fn props(&self) -> &vk::PhysicalDeviceProperties {
        &self.props
    }

    /// The [`vk::PhysicalDeviceMemoryProperties`] of the physical device.
    pub fn mem_props(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.mem_props
    }

    /// The name of the physical device.
    pub fn device_name(&self) -> Cow<str> {
        unsafe { CStr::from_ptr(self.props.device_name.as_ptr()).to_string_lossy() }
    }
}

/// Pick a supported physical device and retrieve its info
///
/// If a device name is provided then its picked by doing a fuzzy search on the
/// list of available devices. Otherwise, the first discrete device is used
pub fn pick_physical_device(
    instance: &Instance,
    instance_exts: &InstanceExts,
    surface: vk::SurfaceKHR,
    device_name: &Option<&str>
) -> Result<(vk::PhysicalDevice, PhysicalDeviceInfo)> {
    // Get available devices
    let phys_devs = unsafe { instance.enumerate_physical_devices() }
        .context("Failed to get available physical devices")?;

    // Shortlist eligible devices
    struct EligibleDevice {
        phys_dev: vk::PhysicalDevice,
        gfx_queue_family: u32,
        props: vk::PhysicalDeviceProperties
    }

    // Check if the device has a queue family supporting graphics operations
    let has_gfx_queue = |&phys_dev| {
        let queue_props = unsafe { instance.get_physical_device_queue_family_properties(phys_dev) };

        queue_props
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(queue_family, _)| (phys_dev, queue_family as u32))
    };

    // Check if the device and graphics queue family support the surface
    let supports_surface = |info: &(vk::PhysicalDevice, u32)| unsafe {
        let (phys_dev, gfx_queue_family) = *info;

        instance_exts
            .surface_ext()
            .get_physical_device_surface_support(phys_dev, gfx_queue_family, surface)
            .eq(&Ok(true))
    };

    // Check if the device supports needed extensions
    let supports_extensions = |info: &(vk::PhysicalDevice, u32)| unsafe {
        let phys_dev = info.0;

        instance
            .enumerate_device_extension_properties(phys_dev)
            .map(|avail_exts| {
                for req_ext in DEVICE_EXTS {
                    let req_ext_name = CStr::from_ptr(req_ext);

                    let found = avail_exts
                        .iter()
                        .map(|avail_ext| CStr::from_ptr(avail_ext.extension_name.as_ptr()))
                        .any(|avail_ext| avail_ext == req_ext_name);

                    if !found {
                        return false;
                    }
                }

                true
            })
            .eq(&Ok(true))
    };

    let mut elig_devs = phys_devs
        .iter()
        .filter_map(has_gfx_queue)
        .filter(supports_surface)
        .filter(supports_extensions)
        .map(|(phys_dev, gfx_queue_family)| {
            let props = unsafe { instance.get_physical_device_properties(phys_dev) };

            EligibleDevice {
                phys_dev,
                gfx_queue_family,
                props
            }
        });

    // Pick a device
    let chosen_dev = match device_name {
        // Fuzzy search the list of eligible devices
        Some(device_name) => {
            let matcher = SkimMatcherV2::default();

            elig_devs
                .filter_map(|elig_dev| {
                    let dev_name = unsafe { CStr::from_ptr(elig_dev.props.device_name.as_ptr()).to_string_lossy() };
                    let score = matcher.fuzzy_match(&dev_name, device_name);

                    score.map(|score| (elig_dev, score))
                })
                .max_by_key(|(_, score)| *score)
                .map(|(elig_dev, _)| elig_dev)
        },

        // Pick first discrete device
        None => elig_devs.find(|elig_dev| elig_dev.props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
    };

    match chosen_dev {
        Some(chosen_dev) => {
            // Get device memory properties
            let mem_props = unsafe { instance.get_physical_device_memory_properties(chosen_dev.phys_dev) };

            let phys_dev_info = PhysicalDeviceInfo {
                gfx_queue_family: chosen_dev.gfx_queue_family,
                props: chosen_dev.props,
                mem_props
            };

            Ok((chosen_dev.phys_dev, phys_dev_info))
        },

        None => bail!("No eligible device found")
    }
}