use std::cmp;

use erupt::{vk, InstanceLoader, DeviceLoader};
use anyhow::{Result, Context};

use super::{
    SURFACE_FORMAT, VkAllocator, MSAALevel, ImageType,
    create_image_views, FramebufferSet,
    name_object, name_multiple
};

use crate::platform::{self, WindowInfo};

const MAX_QUEUED_FRAMES: u32 = 3;

pub struct SyncSet {
    pub swap_image_avail: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub full_frame_finished: vk::Fence
}

pub struct FrameInfo<'a> {
    pub swapchain: vk::SwapchainKHR,
    pub framebuf: vk::Framebuffer,
    pub index: usize,
    pub sync_set: &'a SyncSet
}

pub struct FrameQueue {
    swapchain: vk::SwapchainKHR,
    pub swap_images: Vec<vk::Image>,
    swap_views: Vec<vk::ImageView>,
    pub swap_image_extent: vk::Extent2D,
    framebuf_set: FramebufferSet,
    sync_sets: Vec<SyncSet>,
    frame_index: usize
}

impl FrameQueue {
    pub fn new(
        instance: &InstanceLoader,
        device: &DeviceLoader,
        vk_alloc: &mut VkAllocator,
        window_info: &WindowInfo,
        phys_dev: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        render_pass: vk::RenderPass,
        msaa_level: MSAALevel
    ) -> Result<Self> {
        let capab = unsafe { instance.get_physical_device_surface_capabilities_khr(phys_dev, surface) }
            .result()
            .context("Failed to get device surface capabilities")?;

        // Calculate final swap image extent
        let (width, height) = platform::window_size(window_info)?;

        let extent = if capab.current_extent.width != 0xFFFFFFFF {
            capab.current_extent
        }
        else {
            vk::Extent2D {
                width: cmp::max(
                    capab.min_image_extent.width,
                    cmp::min(capab.max_image_extent.width, width)
                ),
                height: cmp::max(
                    capab.min_image_extent.height,
                    cmp::min(capab.max_image_extent.height, height)
                ),
            }
        };

        // Set number of swapchain images, not exceeding max_image_count
        // Default is MAX_QUEUED_FRAMES
        let num_images = if capab.max_image_count != 0 && MAX_QUEUED_FRAMES > capab.max_image_count {
            capab.max_image_count
        }
        else {
            MAX_QUEUED_FRAMES
        };

        // Create swapchain and get images
        let create_info = vk::SwapchainCreateInfoKHRBuilder::new()
            .surface(surface)
            .min_image_count(num_images)
            .image_format(SURFACE_FORMAT)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capab.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
            .present_mode(vk::PresentModeKHR::IMMEDIATE_KHR)
            .clipped(true);

        let swapchain = unsafe { device.create_swapchain_khr(&create_info, None) }
            .result()
            .context("Failed to create swapchain")?;

        let swap_images = unsafe { device.get_swapchain_images_khr(swapchain, None) }
            .result()
            .context("Failed to get swapchain images")?
            .to_vec();

        name_multiple!(device, swap_images, vk::ObjectType::IMAGE, "Swapchain image");

        let queue_len = swap_images.len();

        // Create swap image views
        let swap_views = create_image_views(device, ImageType::SwapchainImage, &swap_images)?;

        name_multiple!(device, swap_views, vk::ObjectType::IMAGE_VIEW, "Swapchain image view");

        // Create framebuffers
        let framebuf_set = FramebufferSet::new(
            device,
            vk_alloc,
            render_pass,
            msaa_level,
            extent,
            queue_len
        )?;

        // Create sync sets
        let sync_sets = (0..queue_len)
            .map(|i| {
                let semaphore_create_info = vk::SemaphoreCreateInfoBuilder::new();

                let fence_create_info = vk::FenceCreateInfoBuilder::new()
                    .flags(vk::FenceCreateFlags::SIGNALED);

                let swap_image_avail = unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .result()
                    .context("Failed to create swap_image_avail")?;

                name_object(
                    device,
                    vk::ObjectType::SEMAPHORE,
                    swap_image_avail.object_handle(),
                    &format!("swap_image_avail {i}")
                )?;

                let render_finished = unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .result()
                    .context("Failed to create render_finished")?;

                name_object(
                    device,
                    vk::ObjectType::SEMAPHORE,
                    render_finished.object_handle(),
                    &format!("render_finished {i}")
                )?;

                let full_frame_finished = unsafe { device.create_fence(&fence_create_info, None) }
                    .result()
                    .context("Failed to create full_frame_finished")?;

                name_object(
                    device,
                    vk::ObjectType::FENCE,
                    full_frame_finished.object_handle(),
                    &format!("full_frame_finished {i}")
                )?;

                Ok(SyncSet {
                    swap_image_avail,
                    render_finished,
                    full_frame_finished
                })
            })
            .collect::<Result<Vec<SyncSet>>>()
            .context("Failed to create sync sets")?;

        Ok(Self {
            swapchain,
            swap_images,
            swap_views,
            swap_image_extent: extent,
            framebuf_set,
            sync_sets,
            frame_index: 0
        })
    }

    pub fn len(&self) -> usize {
        self.swap_views.len()
    }

    pub fn next_frame(&mut self, device: &DeviceLoader) -> Result<FrameInfo> {
        let sync_set = &self.sync_sets[self.frame_index];

        unsafe {
            // Wait for space in queue to be available so we don't exceed queue_len
            device.wait_for_fences(&[sync_set.full_frame_finished], true, u64::MAX)
                .result()
                .context("Failed to wait for frame_presented")?;

            device.reset_fences(&[sync_set.full_frame_finished])
                .result()
                .context("Failed to reset frame_presented")?;

            // Acquire swapchain image
            device.acquire_next_image_khr(
                self.swapchain,
                u64::MAX,
                sync_set.swap_image_avail,
                vk::Fence::null()
            )
            .result()
            .context("Failed to acquire next swapchain image")?;
        }

        let frame = FrameInfo {
            swapchain: self.swapchain,
            framebuf: self.framebuf_set.framebufs[self.frame_index],
            index: self.frame_index,
            sync_set
        };

        self.frame_index = (self.frame_index + 1) % self.len();

        Ok(frame)
    }

    pub unsafe fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        device.destroy_swapchain_khr(self.swapchain, None);
        
        for &view in &self.swap_views {
            device.destroy_image_view(view, None);
        }

        self.framebuf_set.destroy(device, vk_alloc);
        
        for set in &self.sync_sets {
            device.destroy_semaphore(set.swap_image_avail, None);
            device.destroy_semaphore(set.render_finished, None);
            device.destroy_fence(set.full_frame_finished, None);
        }
    }
}