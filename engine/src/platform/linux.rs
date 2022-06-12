use xcb::{x, Xid};
use anyhow::{Result, Context};

use crate::renderer::Renderer;

pub struct WindowInfo {
    pub conn: xcb::Connection,
    pub window: x::Window
}

pub fn start_engine() -> Result<()> {
    // Start XCB connection
    let (conn, screen_num) = xcb::Connection::connect(None).context("Failed to start XCB connection")?;

    // Get screen and window objects
    let setup = conn.get_setup();

    let screen = setup.roots().nth(screen_num as usize).context("Failed to get screen object")?;
    let window: x::Window = conn.generate_id();

    // Open window
    conn.send_and_check_request(&x::CreateWindow {
        depth: x::COPY_FROM_PARENT as u8,
        wid: window,
        parent: screen.root(),
        x: 0,
        y: 0,
        width: 2048/2,
        height: 1536/2,
        border_width: 0,
        class: x::WindowClass::InputOutput,
        visual: screen.root_visual(),
        value_list: &[
            x::Cw::BackPixel(screen.white_pixel()),
            x::Cw::EventMask(x::EventMask::KEY_PRESS)
        ]
    })
    .context("Failed to open window")?;

    // Get required atoms
    let wm_protocols_cookie = conn.send_request(&x::InternAtom {
        only_if_exists: true,
        name: b"WM_PROTOCOLS"
    });

    let wm_del_window_cookie = conn.send_request(&x::InternAtom {
        only_if_exists: true,
        name: b"WM_DELETE_WINDOW"
    });

    let wm_protocols = conn.wait_for_reply(wm_protocols_cookie)
        .context("Failed to get WM_PROTOCOLS")?
        .atom();

    let wm_del_window = conn.wait_for_reply(wm_del_window_cookie)
        .context("Failed to get WM_DEL_WINDOW")?
        .atom();

    // Enable the window close event
    conn.send_and_check_request(&x::ChangeProperty {
        mode: x::PropMode::Replace,
        window,
        property: wm_protocols,
        r#type: x::ATOM_ATOM,
        data: &[wm_del_window]
    })
    .context("Failed to enable window close event")?;

    // Create renderer
    let window_info = WindowInfo {
        conn,
        window: window.clone()
    };

    // conn will live in window_info this point on
    let conn = &window_info.conn;

    let mut renderer = Renderer::new(&window_info).context("Failed to create renderer")?;

    // Show window
    conn.send_and_check_request(&x::MapWindow { window }).context("Failed to show window")?;

    let start = std::time::Instant::now();
    let mut frames = 0u64;

    // Main game loop
    let res = loop {
        let event_opt = conn.poll_for_event().context("Failed to poll events")?;

        // Handle close event
        if let Some(event) = event_opt {
            if let xcb::Event::X(x::Event::ClientMessage(msg)) = event {
                if let x::ClientMessageData::Data32([atom, ..]) = msg.data() {
                    if atom == wm_del_window.resource_id() {
                        break Ok(());
                    }
                }
            }
        }

        let res = renderer.render();

        if res.is_err() {
            break res;
        }

        frames += 1;
    };

    renderer.destroy();

    let time = std::time::Instant::now().duration_since(start).as_millis() as f32;

    let fps = (frames as f32 / time) * 1000.0;
    let frame_time = time / frames as f32;

    println!("FPS: {fps}");
    println!("Frame Time: {frame_time} ms");

    res
}

pub fn window_size(info: &WindowInfo) -> Result<(u32, u32)> {
    let cookie = info.conn.send_request(&x::GetGeometry { drawable: x::Drawable::Window(info.window) });
    let geom = info.conn.wait_for_reply(cookie).context("Failed to get window size")?;

    Ok((geom.width() as u32, geom.height() as u32))
}