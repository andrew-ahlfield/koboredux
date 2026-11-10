# Audio tools

Tools used to author and check the Halcyon sound theme in
`data/sfx/halcyon`.

## halcyon_music.py

Generates `data/sfx/halcyon/music.a2s`. Each song is described as data
(tempo, chords, bass, arpeggio and drum patterns, melody) and expanded into
plain Audiality 2 script, since A2S has no arrays to sequence from. The
instruments are hand written A2S at the top of the script.

```sh
python3 tools/audio/halcyon_music.py
```

## a2render

Renders one program from an `.a2s` file to a mono WAV, off line, and prints
its peak and RMS levels. Used to check every sound for clipping, silence and
script errors without starting the game.

```sh
cc -O2 -Wall -o a2render tools/audio/a2render.c \
	$(pkg-config --cflags --libs audiality2 2>/dev/null || \
	echo -I/usr/local/include/Audiality2 -L/usr/local/lib -laudiality2) -lm
./a2render data/sfx/halcyon/main.a2s S_ROCK_DEATH 2 rock.wav 0 1 0
```

Arguments after the output file are passed to the program, so
`0 1 0` is (pitch, volume, pan) for an effect.

## spectro.py

Draws the envelope and a log frequency spectrogram of one or more WAV files
into a PNG, which makes timing, decay and clipping problems easy to spot.

```sh
python3 tools/audio/spectro.py out.png rock.wav boom.wav
```

Requires Python 3 with numpy and Pillow.

## Checking a theme in the game

```sh
build/release/src/kobord -data data -sfxtheme 'SFX>>halcyon' \
	-soundtools -st_missing -st_played
```

`-soundtools` logs every slot the theme leaves undefined when it loads;
`-st_missing` logs each time the game asks for one of them.
