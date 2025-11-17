use cgmath::{Deg, Matrix, Matrix4, Point3, Vector3, perspective};
use glfw::{Action, Context, Key};
use rusttype::{Font, Scale, point};

use std::env;
use std::ffi::{CString, c_void};
use std::ptr;

fn compile_shader(src: &str, kind: gl::types::GLenum) -> u32 {
    unsafe {
        let shader = gl::CreateShader(kind);
        let c_src = CString::new(src).unwrap();
        gl::ShaderSource(shader, 1, &c_src.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        let mut success = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success == 0 {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0; len as usize];
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buf.as_mut_ptr() as *mut i8);
            panic!("Shader error: {}", String::from_utf8_lossy(&buf));
        }

        shader
    }
}

fn link_program(vs: u32, fs: u32) -> u32 {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        let mut success = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success == 0 {
            let mut len = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = vec![0; len as usize];
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), buf.as_mut_ptr() as *mut i8);
            panic!("Link error: {}", String::from_utf8_lossy(&buf));
        }

        program
    }
}

/// Build an 8-bit alpha bitmap for the text, automatically scaled so it fits
/// inside a fixed square texture (e.g. 512×512) with a margin, and centered.
fn build_text_bitmap_auto(text: &str) -> (Vec<u8>, u32, u32) {
    static FONT_DATA: &[u8] = include_bytes!("../assets/Arial.ttf");
    let font = Font::try_from_bytes(FONT_DATA).expect("Invalid font");

    const TEX_SIZE: u32 = 512;
    const MARGIN_FRAC: f32 = 0.8;
    const BASE_SCALE_VAL: f32 = 64.0;
    const TEX_W: u32 = TEX_SIZE;
    const TEX_H: u32 = TEX_SIZE;

    // 1) Measure at a base scale.
    let base_scale = Scale::uniform(BASE_SCALE_VAL);
    let base_v_metrics = font.v_metrics(base_scale);

    let base_glyphs: Vec<_> = font
        .layout(text, base_scale, point(0.0, base_v_metrics.ascent))
        .collect();

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for g in &base_glyphs {
        if let Some(bb) = g.pixel_bounding_box() {
            min_x = min_x.min(bb.min.x);
            min_y = min_y.min(bb.min.y);
            max_x = max_x.max(bb.max.x);
            max_y = max_y.max(bb.max.y);
        }
    }

    if min_x == i32::MAX {
        return (Vec::new(), TEX_SIZE, TEX_SIZE);
    }

    let base_w = (max_x - min_x) as f32;
    let base_h = (max_y - min_y) as f32;

    let max_w = TEX_SIZE as f32 * MARGIN_FRAC;
    let max_h = TEX_SIZE as f32 * MARGIN_FRAC;

    let sx = max_w / base_w;
    let sy = max_h / base_h;
    let scale_factor = sx.min(sy);

    let final_scale_val = BASE_SCALE_VAL * scale_factor;
    let final_scale = Scale::uniform(final_scale_val);
    let v_metrics = font.v_metrics(final_scale);

    let glyphs: Vec<_> = font
        .layout(text, final_scale, point(0.0, v_metrics.ascent))
        .collect();

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for g in &glyphs {
        if let Some(bb) = g.pixel_bounding_box() {
            min_x = min_x.min(bb.min.x);
            min_y = min_y.min(bb.min.y);
            max_x = max_x.max(bb.max.x);
            max_y = max_y.max(bb.max.y);
        }
    }

    let text_w = (max_x - min_x) as u32;
    let text_h = (max_y - min_y) as u32;

    let tex_w_i = TEX_W as i32;
    let tex_h_i = TEX_H as i32;

    let x_offset = (tex_w_i - text_w as i32) / 2 - min_x;
    let y_offset = (tex_h_i - text_h as i32) / 2 - min_y;

    let mut bitmap = vec![0u8; (TEX_W * TEX_H) as usize];

    for g in glyphs {
        if let Some(bb) = g.pixel_bounding_box() {
            g.draw(|x, y, v| {
                let x = x as i32 + bb.min.x + x_offset;
                let y = y as i32 + bb.min.y + y_offset;
                if x >= 0 && x < tex_w_i && y >= 0 && y < tex_h_i {
                    let idx = (y as u32 * TEX_W + x as u32) as usize;
                    let val = (v * 255.0) as u8;
                    bitmap[idx] = bitmap[idx].max(val);
                }
            });
        }
    }

    (bitmap, TEX_W, TEX_H)
}

fn create_text_texture(bitmap: &[u8], w: u32, h: u32) -> u32 {
    unsafe {
        let mut tex = 0;
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);

        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RED as i32,
            w as i32,
            h as i32,
            0,
            gl::RED,
            gl::UNSIGNED_BYTE,
            bitmap.as_ptr() as *const _,
        );

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);

        gl::BindTexture(gl::TEXTURE_2D, 0);
        tex
    }
}

fn main() {
    let label = env::args().nth(1).unwrap_or_else(|| "cube".to_string());
    let window_title = format!("cube - {label}");

    let mut glfw = glfw::init(glfw::fail_on_errors).expect("Failed to init GLFW");

    glfw.window_hint(glfw::WindowHint::ContextVersionMajor(3));
    glfw.window_hint(glfw::WindowHint::ContextVersionMinor(3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::Samples(Some(4)));

    let (mut window, events) = glfw
        .create_window(800, 600, &window_title, glfw::WindowMode::Windowed)
        .expect("Failed to create window");

    window.make_current();
    window.set_key_polling(true);
    glfw.set_swap_interval(glfw::SwapInterval::Sync(1));

    gl::load_with(|s| {
        window
            .get_proc_address(s)
            .map(|f| f as *const c_void)
            .unwrap_or(ptr::null())
    });

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gl::Enable(gl::MULTISAMPLE);
    }

    // Cube shaders with text texture
    const CUBE_VS_SRC: &str = r#"
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aUV;

        out vec2 vUV;

        uniform mat4 u_mvp;

        void main() {
            gl_Position = u_mvp * vec4(aPos, 1.0);
            vUV = aUV;
        }
    "#;

    const CUBE_FS_SRC: &str = r#"
        #version 330 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D u_tex;

        void main() {
            float cov = texture(u_tex, vec2(vUV.x, 1.0 - vUV.y)).r;
            vec3 color = mix(vec3(1.0), vec3(0.0, 0.0, 0.0), cov);
            FragColor = vec4(color, 1.0);
        }
    "#;

    let cube_vs = compile_shader(CUBE_VS_SRC, gl::VERTEX_SHADER);
    let cube_fs = compile_shader(CUBE_FS_SRC, gl::FRAGMENT_SHADER);
    let cube_program = link_program(cube_vs, cube_fs);
    unsafe {
        gl::DeleteShader(cube_vs);
        gl::DeleteShader(cube_fs);
    }

    // Edge shaders (black lines)
    const EDGE_VS_SRC: &str = r#"
        #version 330 core
        layout (location = 0) in vec3 aPos;

        uniform mat4 u_mvp;

        void main() {
            gl_Position = u_mvp * vec4(aPos, 1.0);
        }
    "#;

    const EDGE_FS_SRC: &str = r#"
        #version 330 core
        out vec4 FragColor;

        void main() {
            FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    "#;

    let edge_vs = compile_shader(EDGE_VS_SRC, gl::VERTEX_SHADER);
    let edge_fs = compile_shader(EDGE_FS_SRC, gl::FRAGMENT_SHADER);
    let edge_program = link_program(edge_vs, edge_fs);
    unsafe {
        gl::DeleteShader(edge_vs);
        gl::DeleteShader(edge_fs);
    }

    // Cube vertices: position (x,y,z) + uv (u,v)
    const CUBE_VERTICES: [f32; 36 * 5] = [
        // front
        -0.5, -0.5, 0.5, 0.0, 0.0, 0.5, -0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5,
        1.0, 1.0, -0.5, 0.5, 0.5, 0.0, 1.0, -0.5, -0.5, 0.5, 0.0, 0.0, // back
        -0.5, -0.5, -0.5, 1.0, 0.0, 0.5, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, -0.5, 0.0, 1.0, 0.5, 0.5,
        -0.5, 0.0, 1.0, -0.5, 0.5, -0.5, 1.0, 1.0, -0.5, -0.5, -0.5, 1.0, 0.0, // left
        -0.5, 0.5, 0.5, 1.0, 1.0, -0.5, 0.5, -0.5, 0.0, 1.0, -0.5, -0.5, -0.5, 0.0, 0.0, -0.5,
        -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.5, 1.0, 0.0, -0.5, 0.5, 0.5, 1.0, 1.0, // right
        0.5, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5, -0.5, 1.0, 1.0, 0.5, -0.5, -0.5, 1.0, 0.0, 0.5, -0.5,
        -0.5, 1.0, 0.0, 0.5, -0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.0, // top
        -0.5, 0.5, -0.5, 0.0, 1.0, 0.5, 0.5, -0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5,
        0.5, 1.0, 0.0, -0.5, 0.5, 0.5, 0.0, 0.0, -0.5, 0.5, -0.5, 0.0, 1.0, // bottom
        -0.5, -0.5, -0.5, 0.0, 0.0, 0.5, -0.5, -0.5, 1.0, 0.0, 0.5, -0.5, 0.5, 1.0, 1.0, 0.5, -0.5,
        0.5, 1.0, 1.0, -0.5, -0.5, 0.5, 0.0, 1.0, -0.5, -0.5, -0.5, 0.0, 0.0,
    ];

    let mut cube_vao = 0;
    let mut cube_vbo = 0;
    unsafe {
        gl::GenVertexArrays(1, &mut cube_vao);
        gl::GenBuffers(1, &mut cube_vbo);

        gl::BindVertexArray(cube_vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, cube_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (CUBE_VERTICES.len() * std::mem::size_of::<f32>()) as isize,
            CUBE_VERTICES.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        const STRIDE: i32 = (5 * std::mem::size_of::<f32>()) as i32;

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, STRIDE, ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::VertexAttribPointer(
            1,
            2,
            gl::FLOAT,
            gl::FALSE,
            STRIDE,
            (3 * std::mem::size_of::<f32>()) as *const _,
        );
        gl::EnableVertexAttribArray(1);

        gl::BindVertexArray(0);
    }

    // Edge geometry: 8 corners, 12 edges as lines.
    const EDGE_VERTICES: [f32; 8 * 3] = [
        // back face
        -0.5, -0.5, -0.5, // 0
        0.5, -0.5, -0.5, // 1
        0.5, 0.5, -0.5, // 2
        -0.5, 0.5, -0.5, // 3
        // front face
        -0.5, -0.5, 0.5, // 4
        0.5, -0.5, 0.5, // 5
        0.5, 0.5, 0.5, // 6
        -0.5, 0.5, 0.5, // 7
    ];

    const EDGE_INDICES: [u32; 24] = [
        // back square
        0, 1, 1, 2, 2, 3, 3, 0, // front square
        4, 5, 5, 6, 6, 7, 7, 4, // connections
        0, 4, 1, 5, 2, 6, 3, 7,
    ];

    let mut edge_vao = 0;
    let mut edge_vbo = 0;
    let mut edge_ebo = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut edge_vao);
        gl::GenBuffers(1, &mut edge_vbo);
        gl::GenBuffers(1, &mut edge_ebo);

        gl::BindVertexArray(edge_vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, edge_vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (EDGE_VERTICES.len() * std::mem::size_of::<f32>()) as isize,
            EDGE_VERTICES.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, edge_ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (EDGE_INDICES.len() * std::mem::size_of::<u32>()) as isize,
            EDGE_INDICES.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        const EDGE_STRIDE: i32 = (3 * std::mem::size_of::<f32>()) as i32;
        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, EDGE_STRIDE, ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::BindVertexArray(0);
    }

    let cube_u_mvp_loc;
    let cube_u_tex_loc;
    unsafe {
        gl::UseProgram(cube_program);
        cube_u_mvp_loc =
            gl::GetUniformLocation(cube_program, CString::new("u_mvp").unwrap().as_ptr());
        cube_u_tex_loc =
            gl::GetUniformLocation(cube_program, CString::new("u_tex").unwrap().as_ptr());
        gl::Uniform1i(cube_u_tex_loc, 0);
    }

    let edge_u_mvp_loc;
    unsafe {
        gl::UseProgram(edge_program);
        edge_u_mvp_loc =
            gl::GetUniformLocation(edge_program, CString::new("u_mvp").unwrap().as_ptr());
    }

    // Build text texture (auto-scaled and centered).
    let (text_bitmap, text_w, text_h) = build_text_bitmap_auto(&label);
    let text_tex = if text_w > 0 && text_h > 0 {
        create_text_texture(&text_bitmap, text_w, text_h)
    } else {
        0
    };

    let start = std::time::Instant::now();

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) = event {
                window.set_should_close(true);
            }
        }

        let elapsed = start.elapsed().as_secs_f32();
        let (w, h) = window.get_framebuffer_size();

        let proj = perspective(Deg(45.0), w as f32 / h as f32, 0.1, 100.0);
        let view = Matrix4::look_at_rh(
            Point3::new(2.0, 2.0, 3.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let model =
            Matrix4::from_angle_y(Deg(elapsed * 50.0)) * Matrix4::from_angle_x(Deg(elapsed * 30.0));
        let cube_mvp = proj * view * model;

        // Slightly larger model for edges to avoid z-fighting
        let edge_model = model * Matrix4::from_scale(1.001);
        let edge_mvp = proj * view * edge_model;

        unsafe {
            gl::Viewport(0, 0, w, h);
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // Draw cube with text
            gl::UseProgram(cube_program);
            gl::UniformMatrix4fv(cube_u_mvp_loc, 1, gl::FALSE, cube_mvp.as_ptr());

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, text_tex);

            gl::BindVertexArray(cube_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, 36);
            gl::BindVertexArray(0);

            // Draw edges on top
            gl::UseProgram(edge_program);
            gl::UniformMatrix4fv(edge_u_mvp_loc, 1, gl::FALSE, edge_mvp.as_ptr());

            gl::BindVertexArray(edge_vao);
            gl::LineWidth(2.0);
            gl::DrawElements(
                gl::LINES,
                EDGE_INDICES.len() as i32,
                gl::UNSIGNED_INT,
                ptr::null(),
            );
            gl::BindVertexArray(0);
        }

        window.swap_buffers();
        glfw.poll_events();
    }

    // ----- CLEANUP -----

    unsafe {
        gl::DeleteVertexArrays(1, &cube_vao);
        gl::DeleteBuffers(1, &cube_vbo);

        gl::DeleteVertexArrays(1, &edge_vao);
        gl::DeleteBuffers(1, &edge_vbo);
        gl::DeleteBuffers(1, &edge_ebo);

        gl::DeleteTextures(1, &text_tex);
        gl::DeleteProgram(cube_program);
        gl::DeleteProgram(edge_program);
    }

    drop(window);
    drop(glfw);
}
