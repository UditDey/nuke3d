//! Cross platform window functionality

#[cfg(target_os = "linux")]
mod x11;

use ash::vk;
use anyhow::Result;

/// Represents a position in pixels
pub struct Position {
    pub x: u32,
    pub y: u32
}

/// Represents a size in pixels
pub struct Size {
    pub width: u32,
    pub height: u32
}

/// Represents a mouse button
pub enum MouseButton {
    Left,
    Middle,
    Right,
    Other(u8)
}

/// A unique number assigned to each key on the keyboard
pub type Keycode = u32;

/// An event recieved from the window
pub enum WindowEvent {
    KeyPressed(Keycode),
    KeyReleased(Keycode),

    MouseEntered,
    MouseLeft,
    MouseMoved(Position),
    MouseButtonPressed(MouseButton),
    MouseButtonReleased(MouseButton),

    Resized(Size),
    ShouldClose
}

pub enum SurfaceCreateInfo {
    Xlib(vk::XlibSurfaceCreateInfoKHR)
}

/// Represents a window
pub trait Window {
    /// Show/hide the window
    fn set_visible(&self, visible: bool);

    /// Get the window size
    fn size(&self) -> Result<Size>;

    /// Blocks the thread till a new window event is recieved
    fn next_event(&self) -> WindowEvent;

    /// Returns a vulkan XXXSurfaceCreateInfoKHR struct to create a
    /// surface for this window
    fn surface_create_info(&self) -> &SurfaceCreateInfo;
}

/// Create a new window
///
/// Initially in the hidden state, call [`set_visible()`](Window::set_visible()) to show
pub fn create_window(width: u32, height: u32, title: &str) -> Result<Box<dyn Window>> {
    if cfg!(target_os = "linux") {
        let window = x11::X11Window::new(width, height, title)?;

        Ok(Box::new(window))
    }
    else {
        unimplemented!()
    }
}