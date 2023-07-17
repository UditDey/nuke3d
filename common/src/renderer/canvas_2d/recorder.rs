use std::mem;
use std::ffi;
use std::marker::PhantomData;

use vek::{Vec2, Rgba};

#[repr(u32)]
enum CanvasOp {
    StartFill = 0,
    StartStroke = 1,
    LineTo = 2,
    EndContour = 3,
    LastCommand = 4
}

#[repr(C)]
pub struct CanvasCommand {
    opcode: CanvasOp,
    param1: Vec2<u16>,
    param2: Vec2<u16>,
    param3: Vec2<u16>
}

pub(super) const CMD_LIST_BUF_SIZE: u64 = 1000 * mem::size_of::<CanvasCommand>() as u64; // space for 1000 commands

pub struct InitState;
pub struct ContourState;

/// Records canvas commands into the command list buffer
/// 
/// All drawing functions use physical window coordinates with (0, 0) at top left.
/// It is the responsibility of the user to handle DPI scaling, etc
///
/// Uses the typestate pattern to ensure only valid patterns of commands are issued
pub struct Canvas2DRecorder<State> {
    ptr: *mut CanvasCommand,
    _state: PhantomData<State>
}

impl<State> Canvas2DRecorder<State> {
    fn write_cmd(&mut self, cmd: CanvasCommand) {
        unsafe {
            self.ptr.write(cmd);
            self.ptr = self.ptr.add(1);
        }
    }
}

impl Canvas2DRecorder<InitState> {
    pub(super) fn new(cmd_list_ptr: *mut ffi::c_void) -> Self {
        Self {
            ptr: cmd_list_ptr as *mut CanvasCommand,
            _state: PhantomData
        }
    }
    
    pub(super) fn end(mut self) {
        self.write_cmd(CanvasCommand {
            opcode: CanvasOp::LastCommand,
            param1: Vec2::zero(),
            param2: Vec2::zero(),
            param3: Vec2::zero()
        });
    }
    
    pub fn start_fill(mut self, start_point: Vec2<u16>, color: Rgba<u8>) -> Canvas2DRecorder<ContourState> {
        let packed_color = Vec2::new(
            color.r as u16 | (color.g as u16) << 8,
            color.b as u16 | (color.a as u16) << 8
        );
                
        self.write_cmd(CanvasCommand {
            opcode: CanvasOp::StartFill,
            param1: start_point,
            param2: packed_color,
            param3: Vec2::zero()
        });
        
        Canvas2DRecorder {
            ptr: self.ptr,
            _state: PhantomData
        }
    }
    
    pub fn start_stroke(mut self, start_point: Vec2<u16>, color: Rgba<u8>, width: u16) -> Canvas2DRecorder<ContourState> {        
        let packed_color = Vec2::new(
            color.r as u16 | (color.g as u16) << 8,
            color.b as u16 | (color.a as u16) << 8
        );
                
        self.write_cmd(CanvasCommand {
            opcode: CanvasOp::StartStroke,
            param1: start_point,
            param2: packed_color,
            param3: Vec2::new(width, 0)
        });
        
        Canvas2DRecorder {
            ptr: self.ptr,
            _state: PhantomData
        }
    }
}

impl Canvas2DRecorder<ContourState> {
    pub fn line_to(mut self, point: Vec2<u16>) -> Self {
        self.write_cmd(CanvasCommand {
            opcode: CanvasOp::LineTo,
            param1: point,
            param2: Vec2::zero(),
            param3: Vec2::zero()
        });
        
        self
    }
    
    pub fn end(mut self) -> Canvas2DRecorder<InitState> {
        self.write_cmd(CanvasCommand {
            opcode: CanvasOp::EndContour,
            param1: Vec2::zero(),
            param2: Vec2::zero(),
            param3: Vec2::zero()
        });
        
        Canvas2DRecorder {
            ptr: self.ptr,
            _state: PhantomData
        }
    }
}