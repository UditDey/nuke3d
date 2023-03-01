use std::cmp;

use ash::{vk, Device};
use anyhow::{bail, Result, Context};

use crate::window::Window;

use super::{
    instance::InstanceExts,
    device::DeviceExts
};

const DEFAULT_SWAPCHAIN_LEN: u32 = 3;
const SURFACE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

pub struct SyncSet {
    swap_image_avail: vk::Semaphore,
    render_finished: vk::Semaphore,
    queue_submission_finished: vk::Fence
}

impl SyncSet {
    fn new(device: &Device) -> Result<Self> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        let swap_image_avail = unsafe { device.create_semaphore(&semaphore_create_info, None) }
            .context("Failed to create swap_image_avail semaphore")?;

        let render_finished = unsafe { device.create_semaphore(&semaphore_create_info, None) }
            .context("Failed to create render_finished semaphore")?;

        let queue_submission_finished = unsafe { device.create_fence(&fence_create_info, None) }
            .context("Failed to create queue_submission_finished fence")?;

        Ok(Self {
            swap_image_avail,
            render_finished,
            queue_submission_finished
        })
    }

    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.swap_image_avail, None);
            device.destroy_semaphore(self.render_finished, None);
            device.destroy_fence(self.queue_submission_finished, None);
        }
    }

    /// Semaphore which is signalled when the swapchain image is available to render on
    ///
    /// The [`FrameQueue`] takes care of setting up the signalling. Consumer code
    /// should just use it
    pub fn swap_image_avail(&self) -> vk::Semaphore {
        self.swap_image_avail
    }

    /// Semaphore that should be signalled when rendering to the framebuffer has finished
    pub fn render_finished(&self) -> vk::Semaphore {
        self.render_finished
    }

    /// Fence that should be signalled when the frames queue submission has finished
    pub fn queue_submission_finished(&self) -> vk::Fence {
        self.queue_submission_finished
    }
}

/// Objects associated with an acquired frame
pub struct FrameInfo<'a> {
    idx: usize,
    swap_image: vk::Image,
    sync_set: &'a SyncSet
}

impl<'a> FrameInfo<'a> {
    /// The index of this frame
    pub fn idx(&self) -> usize {
        self.idx
    }

    /// The swapchain image to render to in this frame
    pub fn swap_image(&self) -> vk::Image {
        self.swap_image
    }

    /// The sync set for this frame
    pub fn sync_set(&self) -> &SyncSet {
        self.sync_set
    }
}

/// An abstraction over [`vk::SwapchainKHR`] that integrates synchronisation as well
pub struct FrameQueue {
    swapchain: vk::SwapchainKHR,
    swap_images: Vec<vk::Image>,
    sync_sets: Vec<SyncSet>,
    frame_idx: usize
}

impl FrameQueue {
    pub fn new(
        window: &dyn Window,
        instance_exts: &InstanceExts,
        surface: vk::SurfaceKHR,
        phys_dev: vk::PhysicalDevice,
        device: &Device,
        device_exts: &DeviceExts
    ) -> Result<Self> {
        // Get surface capabilities
        let capab = unsafe {
            instance_exts
                .surface_ext()
                .get_physical_device_surface_capabilities(phys_dev, surface)
                .context("Failed to get device surface capabilities")?
        };

        // Calculate swap image extent
        let swap_image_extent = if capab.current_extent.width != u32::MAX {
            capab.current_extent
        }
        else {
            let size = window.size()?;

            vk::Extent2D {
                width: cmp::max(
                    capab.min_image_extent.width,
                    cmp::min(capab.max_image_extent.width, size.width)
                ),
                height: cmp::max(
                    capab.min_image_extent.height,
                    cmp::min(capab.max_image_extent.height, size.height)
                ),
            }
        };

        // Calculate number of swapchain images, ie frame queue length, ie frames in flight
        let num_images = if capab.max_image_count == 0 {
            cmp::max(DEFAULT_SWAPCHAIN_LEN, capab.min_image_count)
        }
        else {
            cmp::min(
                cmp::max(DEFAULT_SWAPCHAIN_LEN, capab.min_image_count),
                capab.max_image_count
            )
        };

        // Create swapchain
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(num_images)
            .image_format(SURFACE_FORMAT)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(swap_image_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capab.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);

        let swapchain = unsafe {
            device_exts
                .swapchain_ext()
                .create_swapchain(&create_info, None)
                .context("Failed to create swapchain")?
        };

        // Get swapchain images
        let swap_images = unsafe {
            device_exts
                .swapchain_ext()
                .get_swapchain_images(swapchain)
                .context("Failed to get swapchain images")?
        };

        let queue_len = swap_images.len();

        // Create sync sets
        let sync_sets = (0..queue_len)
            .map(|_| SyncSet::new(device))
            .collect::<Result<Vec<SyncSet>>>()
            .context("Failed to create sync sets")?;

        Ok(Self {
            swapchain,
            swap_images,
            sync_sets,
            frame_idx: 0
        })
    }

    /// The length of the frame queue (ie number of swapchain images)
    pub fn len(&self) -> usize {
        self.swap_images.len()
    }

    /// Acquire a new frame to render
    ///
    /// This will block the thread till a new frame is available
    pub fn next_frame(&mut self, device: &Device, device_exts: &DeviceExts) -> Result<FrameInfo> {
        let sync_set = &self.sync_sets[self.frame_idx];

        unsafe {
            // Acquire swapchain image
            let (mandated_frame_idx, is_suboptimal) = device_exts
                .swapchain_ext()
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    sync_set.swap_image_avail,
                    vk::Fence::null()
                )
                .context("Failed to acquire next swapchain image")?;

            if is_suboptimal {
                bail!("Suboptimal swapchain image. Handle this case!!");
            }

            if mandated_frame_idx as usize != self.frame_idx {
                bail!("TODO: Do swapchain handling properly");
            }

            device.wait_for_fences(&[sync_set.queue_submission_finished], true, u64::MAX)
                .context("Failed to wait for queue_submission_finished")?;

            device.reset_fences(&[sync_set.queue_submission_finished])
                .context("Failed to reset full_frame_finished")?;
        }

        let frame_info = FrameInfo {
            idx: self.frame_idx,
            swap_image: self.swap_images[self.frame_idx],
            sync_set
        };

        self.frame_idx = (self.frame_idx + 1) % self.len();

        Ok(frame_info)
    }

    pub fn destroy(&self, device: &Device, device_exts: &DeviceExts) {
        unsafe {
            device_exts.swapchain_ext().destroy_swapchain(self.swapchain, None);

            for set in &self.sync_sets {
                set.destroy(device);
            }
        }
    }
}