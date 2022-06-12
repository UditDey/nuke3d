use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context, bail};

use super::{
    MSAALevel, VkAllocator, MemoryBlock, MemoryType,
    RENDER_FORMAT, DEPTH_FORMAT, SURFACE_FORMAT,
    TEXTURE_1_CHANNEL_FORMAT, TEXTURE_4_CHANNEL_FORMAT
};

use crate::nkgui::NKGUI_IMAGE_FORMAT;

#[derive(Clone, Copy, PartialEq)]
pub enum ImageType {
    DepthImage(MSAALevel),
    RenderImage(MSAALevel),
    SwapchainImage,
    FourChannelTexture,
    OneChannelTexture,
    NkGuiImage
}

impl ImageType {
    fn format(&self) -> vk::Format {
        match &self {
            Self::DepthImage(_) => DEPTH_FORMAT,
            Self::RenderImage(_) => RENDER_FORMAT,
            Self::SwapchainImage => SURFACE_FORMAT,
            Self::FourChannelTexture => TEXTURE_4_CHANNEL_FORMAT,
            Self::OneChannelTexture => TEXTURE_1_CHANNEL_FORMAT,
            Self::NkGuiImage => NKGUI_IMAGE_FORMAT,
        }
    }

    fn samples(&self) -> vk::SampleCountFlagBits {
        match &self {
            Self::DepthImage(level) | Self::RenderImage(level) => level.samples(),
            _ => vk::SampleCountFlagBits::_1
        }
    }

    fn usage(&self) -> vk::ImageUsageFlags {
        match &self {
            Self::DepthImage(_) => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            Self::RenderImage(_) | Self::SwapchainImage => vk::ImageUsageFlags::COLOR_ATTACHMENT,
            Self::NkGuiImage => vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            _ => vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
        }
    }

    fn aspect(&self) -> vk::ImageAspectFlags {
        match &self {
            Self::DepthImage(_) => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR
        }
    }
}

pub fn create_image_views(
    device: &DeviceLoader,
    image_type: ImageType,
    images: &[vk::Image]
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|&image| {
            let create_info = vk::ImageViewCreateInfoBuilder::new()
                .image(image)
                .view_type(vk::ImageViewType::_2D)
                .format(image_type.format())
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: image_type.aspect(),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1
                });

            unsafe { device.create_image_view(&create_info, None) }.result()
        })
        .collect::<Result<Vec<vk::ImageView>, vk::Result>>()
        .context("Failed to create image views")
}

pub struct Image {
    image: vk::Image,
    block: MemoryBlock,
    view: vk::ImageView
}

impl Image {
    pub fn new(
        device: &DeviceLoader,
        vk_alloc: &mut VkAllocator,
        image_type: ImageType,
        size: &vk::Extent2D
    ) -> Result<Self> {
        if image_type == ImageType::SwapchainImage {
            bail!("ImageType::SwapchainImage should not be used to create images, only image views");
        }

        // Create image
        let create_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .format(image_type.format())
            .extent(vk::Extent3D { width: size.width, height: size.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(image_type.samples())
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(image_type.usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&create_info, None) }
            .result()
            .context("Failed to create image")?;

        // Get memory requirements
        let req = unsafe { device.get_image_memory_requirements(image) };

        // Allocate and bind memory
        let block = vk_alloc.alloc(device, &req, MemoryType::Device)
            .context("Failed to allocate memory")?;

        unsafe { device.bind_image_memory(image, block.mem(), block.offset()) }
            .result()
            .context("Failed to bind image memory")?;

        // Create image view
        let view = create_image_views(device, image_type, &[image])?[0];

        Ok(Self { image, block, view })
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        };

        vk_alloc.free(self.block);
    }
}