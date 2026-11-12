# Art generators

Every image in `data/gfx/halcyon` is produced by the Python package in
`halcyon/`. Nothing is hand painted, so the theme can be rebuilt, retuned
or extended by editing code and rerunning one command.

## Requirements

- Python 3.9 or newer
- Pillow
- numpy
- scipy

On Debian or Ubuntu: `sudo apt install python3-pil python3-numpy python3-scipy`.

## Rebuilding the theme

```sh
cd tools/art
python3 -m halcyon.build
```

This takes about 45 seconds and overwrites every PNG and the palette in
`data/gfx/halcyon`. `data/gfx/halcyon/main.theme` is hand written and is not
touched. The output is deterministic: the same code always produces the
same pixels, so a rebuild with no source changes leaves `git status` clean.

## How it is put together

| Module | What it makes |
|---|---|
| `palette.py` | The 64 colour master palette (`halcyon.gpl`: 32 base colours plus fire, smoke, plasma, toxic and magenta effect ramps) and the hue shifted material ramps every sprite is shaded with |
| `pix.py` | Hand-pixel style helpers: hue shifted ramp generator (cool shadows, warm highlights) and a mask based RGBA canvas for hard bevels, seams, rivets and lights |
| `render.py` | A small signed distance field renderer: shapes are built from circles, boxes, capsules and polygons, lit from the top left, quantized to a material ramp and outlined. Rotation sheets are rendered by rotating the shape, not the pixels, so every frame is crisp |
| `noise.py` | Seamless value noise and fBm, used for rocks, nebulae and planet surfaces |
| `sprites.py` | Player ship, enemies, motherships, rocks, bullets, bombs, rings, the player's bolts and every explosion (a shared fireball generator gives the blasts a white hot core, smoke, sparks and debris) |
| `tiles.py` | Base tiles for the five regions, drawn with `pix.py`: every cell is a broad hull slab with a south wall; straight cells are low wall walk decks, junctions carry towers with lit windows, ends are turrets and bunkers,
cores are glowing orbs in armoured drums. The 12 and 8 px parallax sets are separate distant silhouettes (flat masses and lights) |
| `backgrounds.py` | The tiling space backdrop, and per region planets and ground layers with two materials each (rock and basalt, dunes and mesas, jungle and rivers, crust and magma fissures, ice and crevasses) plus small outposts |
| `fontdata.py`, `fonts.py` | A hand authored 5x7 font, scaled and styled into the four SoFont sheets |
| `hud.py` | Dashboard frame, LEDs, weapon slots and the title logo |
| `fx.py` | Noise strips, shield effect and the screen transition tiles |
| `build.py` | Calls all of the above and writes the files the theme names |
| `preview.py` | Helper that scales a sheet up for inspection |
| `mazepreview.py` | Builds engine style mazes from the tile sheets over the region ground, at 16, 12 and 8 px, to judge how dense bases read |

Frame layouts (rotation order, tile masks, LED states) follow what the
engine expects; `docs/theme-catalog.md` lists every bank with its frame
count and meaning.

## Art direction

The theme aims for a mid 1990s arcade shooter look: chunky industrial
structures with real mass, multi tone shaded plating with bevels, panel
seams, rivets, vents, hazard striping and glowing lights, a strong top
left light, and fat explosions with smoke and debris. Each region has its
own material: naval gunmetal with cyan lights, desert khaki with rust
trim, olive drab with yellow hazard paint, plum armour with gold trim, and
white ceramic with red markings. Base cores are the one deliberate exception
to the house style: each region's reactor hue appears nowhere else on its
bases, and the core is the biggest, brightest and only large round thing
on a base, so the player's target is always the first thing the eye
lands on.

## Checking the result

Render a scaled preview of any sheet:

```sh
cd tools/art
python3 -m halcyon.preview -s4 player-preview.png ../../data/gfx/halcyon/sprites/player.png
```

Or see the tiles assembled into bases (regions 1 to 5, default all):

```sh
python3 -m halcyon.mazepreview -s 2 maze-preview.png 3
```

Then run the game with the theme forced on, so nothing else can be
mistaken for it:

```sh
build/release/src/kobord -data data -gfxtheme 'GFX>>halcyon' -sfxtheme 'SFX>>halcyon'
```

## License

The generated art is CC BY-SA 4.0 (see `data/gfx/halcyon/LICENSE`). The
scripts in this directory are GPL-2.0-or-later, like the game code.
