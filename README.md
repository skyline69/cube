cube
====

Spin a cube. Stamp it with your word. Enjoy the show.

Works on Linux, macOS and Windows.

# Install it
```bash
cargo install cube
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

License
-------
MIT
