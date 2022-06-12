use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use super::{
    VkAllocator, MSAALevel,
    ImageType, Image, name_multiple
};

pub struct FramebufferSet {
    framebufs: Vec<vk::Framebuffer>,
    render_images: Vec<Image>,
    depth_images: Vec<Image>,
    resolve_images: Vec<Image>,
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
        let render_images = (0..queue_len)
            .map(|_| Image::new(
                device,
                vk_alloc,
                ImageType::RenderImage(msaa_level),
                &size
            ))
            .collect::<Result<Vec<Image>>>()?;

        name_multiple!(
            device,
            render_images.iter().map(|image| image.image()),
            vk::ObjectType::IMAGE,
            "render_images"
        );

        name_multiple!(
            device,
            render_images.iter().map(|image| image.view()),
            vk::ObjectType::IMAGE_VIEW,
            "render_image_views"
        );

        let depth_images = (0..queue_len)
            .map(|_| Image::new(
                device,
                vk_alloc,
                ImageType::DepthImage(msaa_level),
                &size
            ))
            .collect::<Result<Vec<Image>>>()?;

        name_multiple!(
            device,
            depth_images.iter().map(|image| image.image()),
            vk::ObjectType::IMAGE,
            "depth_images"
        );

        name_multiple!(
            device,
            depth_images.iter().map(|image| image.view()),
            vk::ObjectType::IMAGE_VIEW,
            "depth_image_views"
        );

        // If MSAA is enabled, create resolve images
        let resolve_images = if msaa_level != MSAALevel::Off {
            (0..queue_len)
                .map(|_| Image::new(
                    device,
                    vk_alloc,
                    ImageType::RenderImage(MSAALevel::Off),
                    &size
                ))
                .collect::<Result<Vec<Image>>>()?
        }
        else {
            vec![]
        };

        name_multiple!(
            device,
            resolve_images.iter().map(|image| image.image()),
            vk::ObjectType::IMAGE,
            "resolve_images"
        );
        
        name_multiple!(
            device,
            resolve_images.iter().map(|image| image.view()),
            vk::ObjectType::IMAGE_VIEW,
            "resolve_image_views"
        );

        let framebufs = (0..queue_len)
            .map(|i| {
                let mut attachments = vec![render_images[i].view(), depth_images[i].view()];

                if msaa_level != MSAALevel::Off {
                    attachments.push(resolve_images[i].view());
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

        name_multiple!(device, framebufs.iter(), vk::ObjectType::FRAMEBUFFER, "framebufs");

        Ok(FramebufferSet {
            framebufs,
            render_images,
            depth_images,
            resolve_images
        })
    }

    pub fn framebufs(&self) -> &[vk::Framebuffer] {
        self.framebufs.as_slice()
    }

    pub unsafe fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        for &framebuf in &self.framebufs {
            device.destroy_framebuffer(framebuf, None);
        }

        let images = self.render_images
            .into_iter()
            .chain(self.depth_images)
            .chain(self.resolve_images);

        for image in images {
            image.destroy(device, vk_alloc);
        }
    }
}