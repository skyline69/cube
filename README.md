cube
====
![demo](https://github.com/user-attachments/assets/50b58d46-10af-438f-9415-1fa740a3cac9)
> `cube balls`

Spin a cube. Stamp it with your word. Enjoy the show.

Works on Linux, macOS and Windows.

# Install it
```bash
cargo install cube-command
```

Run it
------
```bash
cube "your label"
```


- Label: max 256 chars, defaults to `cube`.
- Esc to quit.

Why it’s cool
-------------
- Auto-scaled text texture so your word always fits.
- Edges stay crisp with a geometry-shader boost (falls back cleanly).
- Simple OpenGL 3.3 pipeline; drop in another font if you like.

Platform requirements
---------------------
- macOS: Xcode Command Line Tools (for clang) and an OpenGL 3.3-capable GPU/driver. GLFW ships prebuilt with the crate—no extra brew install needed.
- Windows: Latest Visual C++ Build Tools or MSVC via Visual Studio; GPU/driver with OpenGL 3.3 support. GLFW is bundled.
- Linux: Build essentials (`clang`/`gcc`, `make`), and GL loader packages (e.g., Ubuntu: `libx11-dev libgl1-mesa-dev libglu1-mesa-dev`). GLFW is bundled.

License
-------
MIT
