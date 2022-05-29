use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use super::{
    VkAllocator, MemoryBlock, MSAALevel,
    ImageType, create_images, create_image_views,
    name_multiple
};

pub struct FramebufferSet {
    pub framebufs: Vec<vk::Framebuffer>,

    render_images: Vec<vk::Image>,
    render_image_mems: Vec<MemoryBlock>,
    render_image_views: Vec<vk::ImageView>,

    depth_images: Vec<vk::Image>,
    depth_image_mems: Vec<MemoryBlock>,
    depth_image_views: Vec<vk::ImageView>,

    resolve_images: Vec<vk::Image>,
    resolve_image_mems: Vec<MemoryBlock>,
    resolve_image_views: Vec<vk::ImageView>
}

impl FramebufferSet {
    pub fn new(
        device: &DeviceLoader,
        vk_alloc: &mut VkAllocator,
        render_pass: vk::RenderPass,
        msaa_level: MSAALevel,
        size: vk::Extent2D,
        queue_len: usize
    ) -> Result<FramebufferSet> {
        // Create render and depth images
        let (render_images, render_image_mems) = create_images(
            device,
            vk_alloc,
            ImageType::RenderImage(msaa_level),
            &vec![size; queue_len]
        )?;

        name_multiple!(device, render_images, vk::ObjectType::IMAGE, "Render image");

        let render_image_views = create_image_views(
            device,
            ImageType::RenderImage(msaa_level),
            &render_images
        )?;

        name_multiple!(device, render_image_views, vk::ObjectType::IMAGE_VIEW, "Render image view");

        let (depth_images, depth_image_mems) = create_images(
            device,
            vk_alloc,
            ImageType::DepthImage(msaa_level),
            &vec![size; queue_len]
        )?;

        name_multiple!(device, depth_images, vk::ObjectType::IMAGE, "Depth image");

        let depth_image_views = create_image_views(
            device,
            ImageType::DepthImage(msaa_level),
            &depth_images
        )?;

        name_multiple!(device, depth_image_views, vk::ObjectType::IMAGE_VIEW, "Depth image view");

        // If MSAA is enabled, create resolve images
        let (resolve_images, resolve_image_mems) = if msaa_level != MSAALevel::Off {
            create_images(
                device,
                vk_alloc,
                ImageType::RenderImage(MSAALevel::Off),
                &vec![size; queue_len]
            )?
        }
        else {
            (vec![], vec![])
        };

        name_multiple!(device, resolve_images, vk::ObjectType::IMAGE, "Resolve image");

        let resolve_image_views = create_image_views(
            device,
            ImageType::RenderImage(MSAALevel::Off),
            &resolve_images
        )?;

        name_multiple!(device, resolve_image_views, vk::ObjectType::IMAGE_VIEW, "Resolve image view");

        let framebufs = (0..queue_len)
            .map(|i| {
                let mut attachments = vec![render_image_views[i], depth_image_views[i]];

                if msaa_level != MSAALevel::Off {
                    attachments.push(resolve_image_views[i]);
                }

                let create_info = vk::FramebufferCreateInfoBuilder::new()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(size.width)
                    .height(size.height)
                    .layers(1);
                
                unsafe { device.create_framebuffer(&create_info, None) }.result()
            })
            .collect::<Result<Vec<vk::Framebuffer>, vk::Result>>()
            .context("Failed to create framebuffers")?;

        name_multiple!(device, framebufs, vk::ObjectType::FRAMEBUFFER, "Framebuffer");

        Ok(FramebufferSet {
            framebufs,
            render_images,
            render_image_mems,
            render_image_views,
            depth_images,
            depth_image_mems,
            depth_image_views,
            resolve_images,
            resolve_image_mems,
            resolve_image_views
        })
    }

    pub unsafe fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        for &framebuf in &self.framebufs {
            device.destroy_framebuffer(framebuf, None);
        }

        let views = self.render_image_views
            .iter()
            .chain(&self.depth_image_views)
            .chain(&self.resolve_image_views);

        for &view in views {
            device.destroy_image_view(view, None);
        }

        let images = self.render_images
            .iter()
            .chain(&self.depth_images)
            .chain(&self.resolve_images);

        for &image in images {
            device.destroy_image(image, None);
        }

        let mems = self.render_image_mems
            .into_iter()
            .chain(self.depth_image_mems)
            .chain(self.resolve_image_mems);

        for mem in mems {
            vk_alloc.free(mem);
        }
    }
}