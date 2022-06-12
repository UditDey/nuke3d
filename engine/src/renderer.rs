use erupt::{vk, EntryLoader, InstanceLoader, DeviceLoader};
use anyhow::{Result, Context};
use piet_gpu::PietGpuRenderContext;

use crate::{
    vk_util::{
        create_instance, create_surface, pick_physical_device,
        create_device, MSAALevel, create_render_pass, FrameQueue, FrameInfo,
        create_command_buffers, VkAllocator, create_image_barrier
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
    _entry: EntryLoader
}

impl Renderer {
    pub fn new(window_info: &WindowInfo) -> Result<Self> {
        let entry = EntryLoader::new().context("Failed to load the Vulkan library")?;
        
        let instance = create_instance(&entry)?;
        let surface = create_surface(&instance, &window_info)?;
        let (phys_dev, phys_dev_info) = pick_physical_device(&instance, surface)?;

        println!("Using device: {}", phys_dev_info.device_name());

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
            frame_queue.swap_image_extent(),
            frame_queue.len(),
            cmd_bufs[0],
            gfx_queue,
            &mut vk_alloc
        ).context("Failed to create NkGuiRenderer")?;

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
            _entry: entry
        })
    }

    pub fn render(&mut self) -> Result<()> {
        // Acquire next frame
        let frame_info = self.frame_queue.next_frame(&self.device)?;
        let cmd_buf = self.cmd_bufs[frame_info.idx()];

        // Record commands
        record_cmds(&self.device, &mut self.nkgui, cmd_buf, &frame_info)?;
        
        // Submit command buffer
        let wait_semaphores = [frame_info.swap_image_avail()];
        let cmd_bufs = [cmd_buf];
        let signal_semaphores = [frame_info.render_finished()];

        let submit_info = vk::SubmitInfoBuilder::new()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device.queue_submit(
                self.gfx_queue,
                &[submit_info],
                frame_info.full_frame_finished()
            )
            .result()
            .context("Failed to submit command buffer")?;
        }

        // Present image
        let wait_semaphores = [frame_info.render_finished()];
        let swapchains = [frame_info.swapchain()];
        let image_indices = [frame_info.idx() as u32];

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
            self.vk_alloc.destroy(&self.device);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_device(None);
            self.instance.destroy_surface_khr(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn record_cmds(
    device: &DeviceLoader,
    nkgui: &mut NkGuiRenderer,
    cmd_buf: vk::CommandBuffer,
    frame_info: &FrameInfo
) -> Result<()> {
    unsafe {
        // Begin command buffer recording
        let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();

        device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
            .result()
            .context("Failed to start command buffer recording")?;

        // nkgui commands
        let mut ctx = PietGpuRenderContext::new();

        nkgui_test(&mut ctx);
        nkgui.cmd_render(device, cmd_buf, &mut ctx, frame_info.idx())?;

        // Blit nkgui render image to swap image
        // Transition swap image to TRANSFER_DST_OPTIMAL for blit
        let barrier = create_image_barrier(
            frame_info.swap_image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL
        );

        device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier]
        );
        
        // Blitting
        let blit_region = vk::ImageBlitBuilder::new()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            })
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: nkgui.render_image_extent().width as i32,
                    y: nkgui.render_image_extent().height as i32,
                    z: 1
                }
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: frame_info.swap_image_extent().width as i32,
                    y: frame_info.swap_image_extent().height as i32,
                    z: 1
                }
            ]);

        device.cmd_blit_image(
            cmd_buf,
            nkgui.render_image(frame_info.idx()),
            vk::ImageLayout::GENERAL,
            frame_info.swap_image(),
            vk::ImageLayout::GENERAL,
            &[blit_region],
            vk::Filter::LINEAR
        );

        // Transition swap image to PRESENT_SRC_KHR for presentation
        let barrier = create_image_barrier(
            frame_info.swap_image(),
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR
        );

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

use piet::{
    RenderContext, kurbo::{Circle, Point, Rect},
    Color, FixedGradient, FixedRadialGradient, GradientStop, Text, TextAttribute, TextLayoutBuilder,
};

fn nkgui_test(rc: &mut impl RenderContext) {
    rc.fill(
        Rect::new(0.0, 0.0, 100.0, 100.0),
        &Color::rgb8(0xBB, 0x2B, 0x68),
    );

    rc.fill(
        Rect::new(100.0, 0.0, 200.0, 100.0),
        &Color::rgb8(0x0A, 0x26, 0xA0),
    );

    rc.fill(
        Rect::new(200.0, 0.0, 300.0, 100.0),
        &Color::rgb8(0x0A, 0xBA, 0x5E),
    );

    rc.fill(
        Rect::new(300.0, 0.0, 400.0, 100.0),
        &Color::rgb8(0x3E, 0xC6, 0xE4),
    );

    let text_size = 50.0;

    rc.save().unwrap();
    //rc.transform(Affine::new([0.2, 0.0, 0.0, -0.2, 200.0, 800.0]));
    let layout = rc
        .text()
        .new_text_layout("Text working??")
        .default_attribute(TextAttribute::FontSize(text_size))
        .build()
        .unwrap();

    rc.draw_text(&layout, Point::new(110.0, 600.0));
    rc.draw_text(&layout, Point::new(210.0, 700.0));
    rc.restore().unwrap();
}