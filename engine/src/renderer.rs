use std::ffi::CStr;

use erupt::{vk, EntryLoader, InstanceLoader, DeviceLoader};
use anyhow::{Result, Context};

use crate::{
    vk_util::{
        create_instance, create_surface, pick_physical_device,
        create_device, MSAALevel, create_render_pass, FrameQueue,
        create_command_buffers, VkAllocator
    },
    platform::WindowInfo,
    nkgui::NkGuiRenderer
};

pub struct Renderer {
    nkgui: NkGuiRenderer,
    cmd_bufs: Vec<vk::CommandBuffer>,
    cmd_pool: vk::CommandPool,
    frame_queue: FrameQueue,
    vk_alloc: VkAllocator,
    render_pass: vk::RenderPass,
    gfx_queue: vk::Queue,
    device: DeviceLoader,
    surface: vk::SurfaceKHR,
    instance: InstanceLoader,
    entry: EntryLoader
}

impl Renderer {
    pub fn new(window_info: &WindowInfo) -> Result<Self> {
        // Create Vulkan objects
        let entry = EntryLoader::new().context("Failed to load the Vulkan library")?;
        
        let instance = create_instance(&entry)?;
        let surface = create_surface(&instance, &window_info)?;
        let (phys_dev, phys_dev_info) = pick_physical_device(&instance, surface)?;

        println!(
            "Using device: {}",
            unsafe { CStr::from_ptr(phys_dev_info.props.device_name.as_ptr()) }.to_string_lossy()
        );

        let (device, gfx_queue) = create_device(&instance, phys_dev, &phys_dev_info)?;

        let msaa_level = MSAALevel::Off;
        let render_pass = create_render_pass(&device, msaa_level)?;

        let mut vk_alloc = VkAllocator::new(&phys_dev_info)?;

        let frame_queue = FrameQueue::new(
            &instance,
            &device,
            &mut vk_alloc,
            window_info,
            phys_dev,
            surface,
            render_pass,
            msaa_level
        )?;

        println!("Frame queue length: {}", frame_queue.len());

        let (cmd_pool, cmd_bufs) = create_command_buffers(&device, frame_queue.len(), &phys_dev_info)?;

        let nkgui = NkGuiRenderer::new(
            &device,
            &phys_dev_info,
            &mut vk_alloc,
            frame_queue.len()
        ).context("Failed to create NkGuiRenderer")?;

        // We pre-record all command buffers
        for (&cmd_buf, &swap_image) in cmd_bufs.iter().zip(&frame_queue.swap_images) {
            record_cmd_buf(&device, cmd_buf, swap_image)?;
        }

        Ok(Self {
            nkgui,
            cmd_bufs,
            cmd_pool,
            frame_queue,
            vk_alloc,
            render_pass,
            gfx_queue,
            device,
            surface,
            instance,
            entry
        })
    }

    pub fn render(&mut self) -> Result<()> {
        let frame_info = self.frame_queue.next_frame(&self.device)?;
        
        // Submit command buffer
        let wait_semaphores = [frame_info.sync_set.swap_image_avail];
        let cmd_bufs = [self.cmd_bufs[frame_info.index]];
        let signal_semaphores = [frame_info.sync_set.render_finished];

        let submit_info = vk::SubmitInfoBuilder::new()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.queue_submit(
                self.gfx_queue,
                &[submit_info],
                frame_info.sync_set.full_frame_finished
            )
            .result()
            .context("Failed to submit command buffer")?;
        }

        // Present image
        let wait_semaphores = [frame_info.sync_set.render_finished];
        let swapchains = [frame_info.swapchain];
        let image_indices = [frame_info.index as u32];

        let present_info = vk::PresentInfoKHRBuilder::new()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.device.queue_present_khr(self.gfx_queue, &present_info)
                .result()
                .context("Failed to present image")?;
        }

        Ok(())
    }

    pub fn destroy(mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.nkgui.destroy(&self.device, &mut self.vk_alloc);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.frame_queue.destroy(&self.device, &mut self.vk_alloc);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            self.instance.destroy_surface_khr(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn record_cmd_buf(device: &DeviceLoader, cmd_buf: vk::CommandBuffer, swap_image: vk::Image) -> Result<()> {
    unsafe {
        // Begin command buffer recording
        let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();

        device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
            .result()
            .context("Failed to start command buffer recording")?;

        // Transition swap image to GENERAL for nkgui compute work
        let barrier = vk::ImageMemoryBarrierBuilder::new()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swap_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS
            });

        device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier]
        );

        // nkgui commands

        // Transition swap image to PRESENT_SRC_KHR for presentation
        let barrier = barrier
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier]
        );

        // End command buffer recording
        device.end_command_buffer(cmd_buf)
            .result()
            .context("Failed to end command buffer recording")?;

        Ok(())
    }
}