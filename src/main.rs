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
    static FONT_DATA: &[u8] = include_bytes!("Arial.ttf");
    let font = Font::try_from_bytes(FONT_DATA).expect("Invalid font");

    // Final texture size used for the cube (square).
    const TEX_SIZE: u32 = 512;
    const MARGIN_FRAC: f32 = 0.8; // text occupies at most 80% of width/height

    // 1) Measure at a base scale.
    let base_scale_val = 64.0;
    let base_scale = Scale::uniform(base_scale_val);
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
        // empty text
        return (Vec::new(), TEX_SIZE, TEX_SIZE);
    }

    let base_w = (max_x - min_x) as f32;
    let base_h = (max_y - min_y) as f32;

    // 2) Compute scale factor so text fits into MARGIN_FRAC of TEX_SIZE.
    let max_w = TEX_SIZE as f32 * MARGIN_FRAC;
    let max_h = TEX_SIZE as f32 * MARGIN_FRAC;

    let sx = max_w / base_w;
    let sy = max_h / base_h;
    let scale_factor = sx.min(sy);

    let final_scale_val = base_scale_val * scale_factor;
    let final_scale = Scale::uniform(final_scale_val);
    let v_metrics = font.v_metrics(final_scale);

    // 3) Layout again at final scale and recompute bounding box.
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

    let tex_w = TEX_SIZE;
    let tex_h = TEX_SIZE;
    let tex_w_i = tex_w as i32;
    let tex_h_i = tex_h as i32;

    // Center the text inside the texture.
    let x_offset = (tex_w_i - text_w as i32) / 2 - min_x;
    let y_offset = (tex_h_i - text_h as i32) / 2 - min_y;

    let mut bitmap = vec![0u8; (tex_w * tex_h) as usize];

    for g in glyphs {
        if let Some(bb) = g.pixel_bounding_box() {
            g.draw(|x, y, v| {
                let x = x as i32 + bb.min.x + x_offset;
                let y = y as i32 + bb.min.y + y_offset;
                if x >= 0 && x < tex_w_i && y >= 0 && y < tex_h_i {
                    let idx = (y as u32 * tex_w + x as u32) as usize;
                    let val = (v * 255.0) as u8;
                    // In case glyphs overlap, take max alpha.
                    bitmap[idx] = bitmap[idx].max(val);
                }
            });
        }
    }

    (bitmap, tex_w, tex_h)
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

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersionMajor(3));
    glfw.window_hint(glfw::WindowHint::ContextVersionMinor(3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let (mut window, events) = glfw
        .create_window(800, 600, &window_title, glfw::WindowMode::Windowed)
        .expect("Failed to create window");

    window.make_current();
    window.set_key_polling(true);

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
    }

    // Cube shaders with text texture
    let cube_vs_src = r#"
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

    let cube_fs_src = r#"
        #version 330 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D u_tex;

        void main() {
            // flip Y so text is upright
            float cov = texture(u_tex, vec2(vUV.x, 1.0 - vUV.y)).r;

            // white background, black text
            vec3 color = mix(vec3(1.0), vec3(0.0, 0.0, 0.0), cov);
            FragColor = vec4(color, 1.0);
        }
    "#;

    let cube_vs = compile_shader(cube_vs_src, gl::VERTEX_SHADER);
    let cube_fs = compile_shader(cube_fs_src, gl::FRAGMENT_SHADER);
    let cube_program = link_program(cube_vs, cube_fs);
    unsafe {
        gl::DeleteShader(cube_vs);
        gl::DeleteShader(cube_fs);
    }

    // Cube vertices: position (x,y,z) + uv (u,v)
    let cube_vertices: [f32; 36 * 5] = [
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
            (cube_vertices.len() * std::mem::size_of::<f32>()) as isize,
            cube_vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        let stride = (5 * std::mem::size_of::<f32>()) as i32;

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, stride, ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::VertexAttribPointer(
            1,
            2,
            gl::FLOAT,
            gl::FALSE,
            stride,
            (3 * std::mem::size_of::<f32>()) as *const _,
        );
        gl::EnableVertexAttribArray(1);

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
        gl::Uniform1i(cube_u_tex_loc, 0); // texture unit 0
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

        unsafe {
            gl::Viewport(0, 0, w, h);
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            gl::UseProgram(cube_program);
            gl::UniformMatrix4fv(cube_u_mvp_loc, 1, gl::FALSE, cube_mvp.as_ptr());

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, text_tex);

            gl::BindVertexArray(cube_vao);
            gl::DrawArrays(gl::TRIANGLES, 0, 36);
            gl::BindVertexArray(0);
        }

        window.swap_buffers();
    }
}
