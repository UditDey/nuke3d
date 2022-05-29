mod linux;
mod windows;

use anyhow::Result;

pub fn start_engine() -> Result<()> {
    #[cfg(unix)]
    return linux::start_engine();

    #[cfg(windows)]
    return windows::start_engine();
}

#[cfg(unix)]
pub type WindowInfo = linux::WindowInfo;

#[cfg(windows)]
pub type WindowInfo = windows::WindowInfo;

pub fn window_size(info: &WindowInfo) -> Result<(u32, u32)> { // (width, height)
    #[cfg(unix)]
    return linux::window_size(info);

    #[cfg(windows)]
    return windows::window_size();
}