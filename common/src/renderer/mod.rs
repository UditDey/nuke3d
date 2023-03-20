//! Renderer for the engine and editor

mod vk_util;
mod canvas_2d;

use ash::{Entry, Instance, Device, vk};
use anyhow::{Result, Context};

use crate::{window::Window, math::Vec2};

use vk_util::{
    instance::{create_instance, InstanceExts},
    surface::create_surface,
    phys_dev::pick_physical_device,
    device::{create_device, DeviceExts},
    frame_queue::FrameQueue,
    cmd_buf::create_command_buffers,
    vma::VmaAllocator
};

use canvas_2d::Canvas2DSystem;

const DEFAULT_FRAMES_IN_FLIGHT: u32 = 2;

/// Renderer configuration
pub struct RendererConfig<'a> {
    pub device_name: Option<&'a str>,
    pub force_validation: bool,
    pub frames_in_flight: Option<u32>
}

/// The nuke3d renderer
pub struct Renderer {
    _entry: Entry,
    instance: Instance,
    instance_exts: InstanceExts,
    surface: vk::SurfaceKHR,
    device: Box<Device>,
    device_exts: DeviceExts,
    gfx_queue: vk::Queue,
    frame_queue: FrameQueue,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    vma_alloc: VmaAllocator,
    canvas_2d_sys: Canvas2DSystem
}

impl Renderer {
    pub fn new(config: &RendererConfig, window: &dyn Window) -> Result<Self> {
        // Load vulkan
        let entry = unsafe { Entry::load().context("Failed to load vulkan")? };

        // Create vulkan objects
        let (instance, instance_exts) = create_instance(&entry, window, config.force_validation)?;
        let surface = create_surface(&instance_exts, window)?;
        let (phys_dev, phys_dev_info) = pick_physical_device(&instance, &instance_exts, surface, &config.device_name)?;

        println!("Using device: {}", phys_dev_info.device_name());

        let (device, device_exts, gfx_queue) = create_device(&instance, phys_dev, &phys_dev_info)?;
        
        let frames_in_flight = config.frames_in_flight.unwrap_or(DEFAULT_FRAMES_IN_FLIGHT);
        
        let frame_queue = FrameQueue::new(
            window,
            &instance_exts,
            surface,
            phys_dev,
            &device,
            &device_exts,
            frames_in_flight
        )?;

        let (cmd_pool, cmd_bufs) = create_command_buffers(&device, &phys_dev_info, frames_in_flight)?;
        let vma_alloc = VmaAllocator::new(&instance, phys_dev, &device)?;
        
        let canvas_2d_sys = Canvas2DSystem::new(&device, &frame_queue, &vma_alloc, frames_in_flight)?;
        
        println!("Number of swapchain images: {}", frame_queue.swap_image_views().len());
        println!("Frames in flight: {frames_in_flight}");

        Ok(Self {
            _entry: entry,
            instance,
            instance_exts,
            surface,
            device,
            device_exts,
            gfx_queue,
            frame_queue,
            cmd_pool,
            cmd_bufs,
            vma_alloc,
            canvas_2d_sys
        })
    }

    /// Render a frame
    ///
    /// This function blocks till a new frame is available to render
    pub fn render_frame(&mut self) -> Result<()> {
        unsafe {
            // Wait to acquire new frame
            let frame_info = self.frame_queue.next_frame(&self.device, &self.device_exts)?;
    
            // Begin command buffer recording
            let cmd_buf = self.cmd_bufs[frame_info.frame_idx()];
            let begin_info = vk::CommandBufferBeginInfo::default();
    
            self.device
                .begin_command_buffer(cmd_buf, &begin_info)
                .context("Failed to begin command buffer recording")?;
    
            // Transition swapchain image layout from UNDEFINED to GENERAL
            let barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(frame_info.swap_image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
                
            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier]
            );
            
            // Record canvas2d commands
            let mut canvas_2d = self.canvas_2d_sys.new_canvas(&frame_info);
            
            canvas_2d.move_to(Vec2::new(200.0, 200.0));
            canvas_2d.line_to(Vec2::new(400.0, 200.0));
            canvas_2d.line_to(Vec2::new(300.0, 300.0));
            canvas_2d.line_to(Vec2::new(400.0, 400.0));
            canvas_2d.line_to(Vec2::new(200.0, 400.0));
            canvas_2d.line_to(Vec2::new(200.0, 200.0));
            
            canvas_2d.move_to(Vec2::new(500.0, 450.0));
            canvas_2d.line_to(Vec2::new(650.0, 400.0));
            canvas_2d.line_to(Vec2::new(512.0, 498.0));
            canvas_2d.line_to(Vec2::new(687.0, 340.0));
            canvas_2d.line_to(Vec2::new(565.0, 400.0));
            canvas_2d.line_to(Vec2::new(500.0, 450.0));
            
            self.canvas_2d_sys.draw(&self.device, cmd_buf, &frame_info, canvas_2d);
            
            // Transition swapchain image layout from GENERAL TO PRESENT_SRC_KHR
            let barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(frame_info.swap_image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
                
            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier]
            );
            
            // End command buffer recording
            self.device
                .end_command_buffer(cmd_buf)
                .context("Failed to end command buffer recording")?;
                
            // Submit command buffer
            let wait_semaphores = [frame_info.sync_set().swap_image_avail()];
            let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
            
            let cmd_bufs = [cmd_buf];
            let signal_semaphores = [frame_info.sync_set().cmd_buf_done()];
            
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(&signal_semaphores)
                .build();
                
            self.device
                .queue_submit(self.gfx_queue, &[submit_info], frame_info.sync_set().frame_done())
                .context("Failed to submit command buffer")?;
    
            // Present frame        
            let wait_semaphores = [frame_info.sync_set().cmd_buf_done()];
            let swapchains = [frame_info.swapchain()];
            let image_indices = [frame_info.swap_image_idx() as u32];
            
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            
            self.device_exts
                .swapchain_ext()
                .queue_present(self.gfx_queue, &present_info)
                .context("Failed to present frame")?;
            
            Ok(())
        }
    }

    pub fn destroy(self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.canvas_2d_sys.destroy(&self.device, &self.vma_alloc);
            self.vma_alloc.destroy();
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.frame_queue.destroy(&self.device, &self.device_exts);
            self.device.destroy_device(None);
            self.instance_exts.surface_ext().destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}