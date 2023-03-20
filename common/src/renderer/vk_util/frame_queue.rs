use std::cmp;

use ash::{vk, Device};
use anyhow::{bail, Result, Context};

use crate::window::Window;

use super::{
    instance::InstanceExts,
    device::DeviceExts,
};

const SURFACE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Synchronization objects for a frame
pub struct SyncSet {
    swap_image_avail: vk::Semaphore,
    cmd_buf_done: vk::Semaphore,
    frame_done: vk::Fence
}

impl SyncSet {
    fn new(device: &Device) -> Result<Self> {
        unsafe {
            let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            
            let swap_image_avail = device.create_semaphore(&semaphore_create_info, None)
                .context("Failed to create semaphore")?;
            
            let cmd_buf_done = device.create_semaphore(&semaphore_create_info, None)
                .context("Failed to create semaphore")?;
                
            let frame_done = device.create_fence(&fence_create_info, None)
                .context("Failed to create fence")?;
                
            Ok(Self {
                swap_image_avail,
                cmd_buf_done,
                frame_done
            })
        }
    }
    
    /// Semaphore signalled when the swapchain image (and corresponding framebuffer)
    /// is available
    ///
    /// The [`FrameQueue`] takes are of signalling this. Command buffers should wait on this
    /// semaphore before any operations/pipeline stages that write to the swapchain image
    /// or framebuffer
    pub fn swap_image_avail(&self) -> vk::Semaphore {
        self.swap_image_avail
    }
    
    /// Semaphore that should be signalled when the frame's command buffer has finished
    ///
    /// This should be signalled in [`SubmitInfo`](vk::SubmitInfo) and this waited on
    /// before frame presentation
    pub fn cmd_buf_done(&self) -> vk::Semaphore {
        self.cmd_buf_done
    }
    
    /// Fence that should be signalled when the frame's queue submission has finished
    ///
    /// This should be signalled by [`Device::queue_submit()`]
    pub fn frame_done(&self) -> vk::Fence {
        self.frame_done
    }
    
    fn destroy(self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.swap_image_avail, None);
            device.destroy_semaphore(self.cmd_buf_done, None);
            device.destroy_fence(self.frame_done, None);
        }
    }
}

/// Objects associated with an acquired frame
pub struct FrameInfo<'a> {
    frame_idx: usize,
    swap_image_idx: usize,
    swapchain: vk::SwapchainKHR,
    sync_set: &'a SyncSet,
    swap_image: vk::Image,
    swap_image_extent: &'a vk::Extent2D
}

impl<'a> FrameInfo<'a> {
    /// This frame's index
    ///
    /// This should be used to select per flight in frame resources such as command buffers
    pub fn frame_idx(&self) -> usize {
        self.frame_idx
    }
    
    /// This frame's swapchain image's index
    ///
    /// This is useful while presenting the frame and also to select descriptor sets that
    /// point to swapchain images
    pub fn swap_image_idx(&self) -> usize {
        self.swap_image_idx
    }
    
    /// The swapchain that should be used while presenting the frame
    pub fn swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }
    
    /// This frame's [`SyncSet`]
    pub fn sync_set(&self) -> &SyncSet {
        self.sync_set
    }
    
    /// The swapchain image to draw to
    pub fn swap_image(&self) -> vk::Image {
        self.swap_image
    }
    
    /// The size of the swapchain image to draw to
    pub fn swap_image_extent(&self) -> &vk::Extent2D {
        self.swap_image_extent
    }
}

/// An abstraction over [`vk::SwapchainKHR`] that integrates frame synchronization as well
pub struct FrameQueue {
    swapchain: vk::SwapchainKHR,
    swap_image_extent: vk::Extent2D,
    swap_images: Vec<vk::Image>,
    swap_image_views: Vec<vk::ImageView>,
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
        device_exts: &DeviceExts,
        frames_in_flight: u32
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
        
        // Calculate number of swapchain images
        // Must be more than minimum, less than maximum, and atleast frames_in_flight
        let num_images = if capab.max_image_count == 0 {
            // No maximum
            cmp::max(capab.min_image_count, frames_in_flight)
        }
        else {
            if frames_in_flight > capab.max_image_count {
                bail!("Number of frames in flight greater than max swapchain images");
            }
            
            cmp::min(
                cmp::max(capab.min_image_count, frames_in_flight),
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
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
        
        // Create swapchain image views
        let swap_image_views = swap_images
            .iter()
            .map(|&image| unsafe {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(SURFACE_FORMAT)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1
                    });
    
                device.create_image_view(&create_info, None)
            })
            .collect::<Result<Vec<vk::ImageView>, vk::Result>>()
            .context("Failed to create swap image views")?;
        
        // Create sync sets         
        let sync_sets = (0..frames_in_flight)
            .map(|_| SyncSet::new(device))
            .collect::<Result<Vec<SyncSet>>>()?;
                    
        Ok(Self {
            swapchain,
            swap_image_extent,
            swap_images,
            swap_image_views,
            sync_sets,
            frame_idx: 0
        })
    }
    
    pub fn swap_image_views(&self) -> &[vk::ImageView] {
        self.swap_image_views.as_slice()
    }
    
    pub fn next_frame(&mut self, device: &Device, device_exts: &DeviceExts) -> Result<FrameInfo> {        
        unsafe {
            let sync_set = &self.sync_sets[self.frame_idx];
            
            // Acquire swapchain image
            let (swap_image_idx, is_suboptimal) = device_exts
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
         
            // Wait for previous frame in this slot to finish
            device.wait_for_fences(&[sync_set.frame_done], true, u64::MAX)
                .context("Failed to wait for frame_done fence")?;
                  
            device.reset_fences(&[sync_set.frame_done])
                .context("Failed to reset frame_done fence")?;
                
            let info = FrameInfo {
                frame_idx: self.frame_idx,
                swap_image_idx: swap_image_idx as usize,
                swapchain: self.swapchain,
                sync_set,
                swap_image: self.swap_images[self.frame_idx],
                swap_image_extent: &self.swap_image_extent
            };
            
            let frames_in_flight = self.sync_sets.len();
            self.frame_idx = (self.frame_idx + 1) % frames_in_flight;

            Ok(info)
        }
    }
    
    pub fn destroy(self, device: &Device, device_exts: &DeviceExts) {
        unsafe {
            device_exts.swapchain_ext().destroy_swapchain(self.swapchain, None);
            
            for view in self.swap_image_views {
                device.destroy_image_view(view, None);
            }
            
            for set in self.sync_sets {
                set.destroy(device);
            }
        }
    }
}