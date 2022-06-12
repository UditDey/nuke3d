use std::cmp;

use erupt::{vk, InstanceLoader, DeviceLoader};
use anyhow::{Result, Context};

use super::{
    SURFACE_FORMAT, VkAllocator, MSAALevel, ImageType,
    create_image_views, FramebufferSet, name_multiple
};

use crate::platform::{self, WindowInfo};

const MAX_QUEUED_FRAMES: u32 = 3;

pub struct SyncSet {
    swap_image_avail: vk::Semaphore,
    render_finished: vk::Semaphore,
    full_frame_finished: vk::Fence
}

pub struct FrameInfo<'a> {
    swapchain: vk::SwapchainKHR,
    swap_image: vk::Image,
    swap_image_extent: vk::Extent2D,
    framebuf: vk::Framebuffer,
    idx: usize,
    sync_set: &'a SyncSet
}

impl<'a> FrameInfo<'a> {
    pub fn swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn swap_image(&self) -> vk::Image {
        self.swap_image
    }

    pub fn swap_image_extent(&self) -> &vk::Extent2D {
        &self.swap_image_extent
    }

    pub fn framebuf(&self) -> vk::Framebuffer {
        self.framebuf
    }

    pub fn idx(&self) -> usize {
        self.idx
    }

    pub fn swap_image_avail(&self) -> vk::Semaphore {
        self.sync_set.swap_image_avail
    }

    pub fn render_finished(&self) -> vk::Semaphore {
        self.sync_set.render_finished
    }

    pub fn full_frame_finished(&self) -> vk::Fence {
        self.sync_set.full_frame_finished
    }
}

pub struct FrameQueue {
    swapchain: vk::SwapchainKHR,
    swap_images: Vec<vk::Image>,
    swap_views: Vec<vk::ImageView>,
    swap_image_extent: vk::Extent2D,
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

        name_multiple!(device, swap_images.iter(), vk::ObjectType::IMAGE, "swap_image");

        let queue_len = swap_images.len();

        // Create swap image views
        let swap_views = create_image_views(device, ImageType::SwapchainImage, &swap_images)?;

        name_multiple!(device, swap_views.iter(), vk::ObjectType::IMAGE_VIEW, "swap_view");

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
            .map(|_| {
                let semaphore_create_info = vk::SemaphoreCreateInfoBuilder::new();

                let fence_create_info = vk::FenceCreateInfoBuilder::new()
                    .flags(vk::FenceCreateFlags::SIGNALED);

                let swap_image_avail = unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .result()
                    .context("Failed to create swap_image_avail")?;

                let render_finished = unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .result()
                    .context("Failed to create render_finished")?;

                let full_frame_finished = unsafe { device.create_fence(&fence_create_info, None) }
                    .result()
                    .context("Failed to create full_frame_finished")?;

                Ok(SyncSet {
                    swap_image_avail,
                    render_finished,
                    full_frame_finished
                })
            })
            .collect::<Result<Vec<SyncSet>>>()
            .context("Failed to create sync sets")?;

        name_multiple!(
            device,
            sync_sets.iter().map(|set| set.swap_image_avail),
            vk::ObjectType::SEMAPHORE,
            "swap_image_avail"
        );

        name_multiple!(
            device,
            sync_sets.iter().map(|set| set.render_finished),
            vk::ObjectType::SEMAPHORE,
            "render_finished"
        );

        name_multiple!(
            device,
            sync_sets.iter().map(|set| set.full_frame_finished),
            vk::ObjectType::FENCE,
            "full_frame_finished"
        );

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

    pub fn swap_image_extent(&self) -> vk::Extent2D {
        self.swap_image_extent.clone()
    }

    pub fn swap_image_views(&self) -> &[vk::ImageView] {
        self.swap_views.as_slice()
    }

    pub fn len(&self) -> usize {
        self.swap_images.len()
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
            swap_image: self.swap_images[self.frame_index],
            swap_image_extent: self.swap_image_extent.clone(),
            framebuf: self.framebuf_set.framebufs()[self.frame_index],
            idx: self.frame_index,
            sync_set
        };

        self.frame_index = (self.frame_index + 1) % self.len();

        Ok(frame)
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        unsafe{
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
}