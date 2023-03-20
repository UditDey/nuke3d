use std::mem;

use super::Metadata;

use crate::{
    math::Vec2,
    renderer::vk_util::buffer::TransferBuffer
};

/// Represents a 2D vector graphics canvas
///
/// Drawing is done with standard vector graphics elements:
/// lines, quadratic bezier curves and cubic bezier curves.
/// All curves are drawn relative to an internal cursor position.
///
/// Note that it is the user's responsibility to ensure that
/// boundaries form closed loops
pub struct Canvas2D<'a> {
    pub(super) metadata: Metadata,
    lines_buf: &'a TransferBuffer,
    offset: usize,
    cursor_pos: Vec2
}

impl<'a> Canvas2D<'a> {
    pub fn new(lines_buf: &'a TransferBuffer) -> Self {
        Self {
            metadata: Metadata { num_lines: 0 },
            lines_buf,
            offset: 0,
            cursor_pos: Vec2::zero()
        }
    }

    /// Moves the cursor to a given position
    pub fn move_to(&mut self, pos: Vec2) {
        self.cursor_pos = pos;
    }
    
    /// Mark a boundary line from the cursor position to the given point
    pub fn line_to(&mut self, point: Vec2) {
        // Check if enough space is left in lines buffer
        let space_req = 2 * mem::size_of::<Vec2>();
        let space_left = self.lines_buf.size() as usize - self.offset;

        if space_left < space_req {
            panic!("Ran out of space in lines buffer");
        }

        unsafe {
            let ptr = self.lines_buf.ptr().add(self.offset) as *mut Vec2;

            ptr.write(self.cursor_pos);
            ptr.add(1).write(point);
        }

        self.metadata.num_lines += 1;
        self.offset += space_req;
        self.cursor_pos = point;
    }
}