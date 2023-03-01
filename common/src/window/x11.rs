//! Functionality for creating X11 windows on linux using Xlib

use std::ptr;
use std::mem;
use std::os;
use std::ffi::{self, CString};

use ash::vk;
use x11_dl::xlib;
use anyhow::{bail, Result, Context};

use super::{Window, WindowEvent, MouseButton, Position, Size, SurfaceCreateInfo};

pub struct X11Window {
    xlib: xlib::Xlib,
    display: *mut xlib::_XDisplay,
    window: xlib::Window,
    wm_protocols: xlib::Atom,
    wm_delete_window: xlib::Atom,
    surface_create_info: SurfaceCreateInfo
}

impl X11Window {
    pub fn new(width: u32, height: u32, title: &str) -> Result<Self> {
        unsafe {
            // Load Xlib
            let xlib = xlib::Xlib::open().context("Failed to load Xlib")?;

            // Enable Xlib threading
            // Both SDL2 and vkcube example do this, allegedly this is needed for
            // threading within the nvidia propreitary driver however not sure if
            // it's relevant to us or only for opengl, lets do this anyways
            (xlib.XInitThreads)();

            // Open display connection
            let display = (xlib.XOpenDisplay)(ptr::null());

            if display.is_null() {
                bail!("Failed to open display connection");
            }

            // Create window
            let screen = (xlib.XDefaultScreen)(display);
            let root = (xlib.XRootWindow)(display, screen);

            // Window attributes
            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();

            // Black background
            attributes.background_pixel = (xlib.XBlackPixel)(display, screen);

            // Events we're listening for
            attributes.event_mask =
                xlib::KeyPressMask |
                xlib::KeyReleaseMask |
                xlib::EnterWindowMask |
                xlib::LeaveWindowMask |
                xlib::PointerMotionMask |
                xlib::ButtonPressMask |
                xlib::ButtonReleaseMask |
                xlib::ResizeRedirectMask;

            let attributes_mask = xlib::CWBackPixel | xlib::CWEventMask;

            let window = (xlib.XCreateWindow)(
                display,
                root,
                0, 0,
                width, height,
                0,
                0,
                xlib::InputOutput as ffi::c_uint,
                ptr::null_mut(),
                attributes_mask,
                &mut attributes
            );

            // Set window title
            let title = CString::new(title).context("Invalid window title")?;

            (xlib.XStoreName)(display, window, title.as_ptr() as *mut os::raw::c_char);

            // Intern needed atoms
            let wm_protocols_str = CString::new("WM_PROTOCOLS").unwrap();
            let wm_delete_window_str = CString::new("WM_DELETE_WINDOW").unwrap();

            let wm_protocols = (xlib.XInternAtom)(display, wm_protocols_str.as_ptr(), xlib::False);
            let wm_delete_window = (xlib.XInternAtom)(display, wm_delete_window_str.as_ptr(), xlib::False);

            // Hook close request
            let mut protocols = [wm_delete_window];

            (xlib.XSetWMProtocols)(display, window, protocols.as_mut_ptr(), protocols.len() as ffi::c_int);

            // Flush connection for good measure
            (xlib.XFlush)(display);

            // Vulkan surface create info
            let dpy: *mut vk::Display = mem::transmute(display);

            let surface_create_info = SurfaceCreateInfo::Xlib(
                vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(dpy)
                    .window(window)
                    .build()
            );

            Ok(Self {
                xlib,
                display,
                window,
                wm_protocols,
                wm_delete_window,
                surface_create_info
            })
        }
    }
}

impl Window for X11Window {
    fn set_visible(&self, visible: bool) {
        unsafe {
            if visible {
                (self.xlib.XMapWindow)(self.display, self.window);
            }
            else {
                (self.xlib.XUnmapWindow)(self.display, self.window);
            }

            (self.xlib.XFlush)(self.display);
        }
    }

    fn size(&self) -> Result<Size> {
        unsafe {
            let mut ret_window: xlib::Window = mem::zeroed();
            let mut x = 0i32;
            let mut y = 0i32;
            let mut width = 0u32;
            let mut height = 0u32;
            let mut border_width = 0u32;
            let mut depth = 0u32;

            let status = (self.xlib.XGetGeometry)(
                self.display,
                self.window,
                &mut ret_window,
                &mut x, &mut y,
                &mut width, &mut height,
                &mut border_width,
                &mut depth
            );

            if status == 0 {
                bail!("Failed to get window geometry");
            }

            Ok(Size { width, height })
        }
    }

    fn next_event(&self) -> WindowEvent {
        unsafe {
            // Keep consuming events till a relevant event is recieved
            let mut event: xlib::XEvent = mem::zeroed();

            loop {
                (self.xlib.XNextEvent)(self.display, &mut event);

                match event.get_type() {
                    // Key pressed
                    xlib::KeyPress => {
                        let event = xlib::XKeyPressedEvent::from(event);

                        break WindowEvent::KeyPressed(event.keycode);
                    },

                    // Key released
                    xlib::KeyRelease => {
                        let event = xlib::XKeyReleasedEvent::from(event);

                        break WindowEvent::KeyReleased(event.keycode);
                    },

                    // Mouse entered
                    xlib::EnterNotify => break WindowEvent::MouseEntered,

                    // Mouse left
                    xlib::LeaveNotify => break WindowEvent::MouseLeft,

                    // Mouse moved
                    xlib::MotionNotify => {
                        let event = xlib::XMotionEvent::from(event);

                        break WindowEvent::MouseMoved(Position { x: event.x as u32, y: event.y as u32 });
                    },

                    // Mouse button pressed
                    xlib::ButtonPress => {
                        let event = xlib::XButtonPressedEvent::from(event);

                        break WindowEvent::MouseButtonPressed(map_mouse_button(event.button));
                    },

                    // Mouse button pressed
                    xlib::ButtonRelease => {
                        let event = xlib::XButtonReleasedEvent::from(event);

                        break WindowEvent::MouseButtonReleased(map_mouse_button(event.button));
                    },

                    // Window resized
                    xlib::ResizeRequest => {
                        let event = xlib::XResizeRequestEvent::from(event);

                        break WindowEvent::Resized(Size { width: event.width as u32, height: event.height as u32 })
                    }

                    // Client message
                    xlib::ClientMessage => {
                        let event = xlib::XClientMessageEvent::from(event);

                        if event.message_type == self.wm_protocols && event.format == 32 {
                            let protocol = event.data.get_long(0) as xlib::Atom;

                            if protocol == self.wm_delete_window {
                                break WindowEvent::ShouldClose;
                            }
                        }
                    },

                    _ => ()
                }
            }
        }
    }

    fn surface_create_info(&self) -> &SurfaceCreateInfo {
        &self.surface_create_info
    }
}

fn map_mouse_button(button: os::raw::c_uint) -> MouseButton {
    match button {
        1 => MouseButton::Left,
        2 => MouseButton::Middle,
        3 => MouseButton::Right,
        other => MouseButton::Other(other as u8)
    }
}