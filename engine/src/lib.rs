mod platform;
mod vk_util;
mod renderer;
mod nkgui;

use anyhow::Result;

pub fn start_engine() -> Result<()> {
    platform::start_engine()
}