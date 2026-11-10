"""Generate data/sfx/halcyon/music.a2s, the Halcyon soundtrack.

Usage: python3 tools/audio/halcyon_music.py

Audiality 2 scripts have no arrays, so writing a soundtrack by hand means
spelling out every note as a timed instruction. Instead, each song is
described here as data (tempo, chord progression, bass, arpeggio and drum
patterns, melody) and this script expands it into plain A2S. The
instruments themselves are hand written A2S in INSTRUMENTS below.

Copyright 2026 Andrew Ahlfield. The generated music is CC BY-SA 4.0, like
the rest of the Halcyon sound theme.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "data", "sfx", "halcyon", "music.a2s")

HEADER = '''def title	"HalcyonMusic"
def version	"1.0.0"
def description	"Kobo Redux Halcyon Sound Theme: Music"
def author	"Andrew Ahlfield"
def copyright	"Copyright 2026 Andrew Ahlfield"
def license	"CC BY-SA 4.0"
def a2sversion	"1.9.1"

// GENERATED FILE: edit tools/audio/halcyon_music.py and rerun it.
//
// Pitch 0 is middle C. Instruments take a pitch, a volume, a length in ms
// where it matters, and a pan position.
'''

INSTRUMENTS = r'''

  ///////////////////////////////////////////////////////////////////////////
 // Instruments
///////////////////////////////////////////////////////////////////////////

// Plucked saw bass: the filter snaps shut over the first part of the note
MBass(P V=1 L=200)
{
	struct { wtosc; filter12; panmix }
	w saw; @p P; @cutoff (P + 4); q .15
	@a (V * .32)
	cutoff (P + 1.5); d (L * .6)
	a (V * .2); d (L * .4)
	a 0; d 40
}

// Glassy FM pluck for arpeggios
MPluck(P V=1 X=0)
{
	struct { fm2; panmix }
	pan X; set
	@p P; @p1 1; @a1 .9; @a (V * .16)
	a1 .08; a (V * .05); d 110
	a 0; d 260
}

// One pad voice: a lowpassed saw that swells in and fades out
MPadVoice(P V=1 L=1000 X=0)
{
	struct { wtosc; filter12; panmix }
	pan X; set
	w saw; phase (rand 1); @p P; @cutoff (P + 1); q .05
	@a 0
	a (V * .05); cutoff (P + 2.2); d (L * .35)
	cutoff (P + 1.6); d (L * .45)
	a 0; d (L * .45)
}

// A pad chord note: two voices, slightly detuned and spread apart
MPad(P V=1 L=1000 X=0)
{
	MPadVoice (P + .004) V L (X - .4)
	MPadVoice (P - .004) V L (X + .4)
}

// Warm FM lead with delayed vibrato
MLead(P V=1 L=400)
{
	struct { fm2; panmix }
	pan .15; set
	@p P; @p1 0; @a1 1.1; @fb .15; @a 0
	a (V * .2); a1 .55; d 25
	!t 25
	!v 0
	while t < L {
		p (P + v); d 55
		p (P - v); d 55
		+t 110
		if v < .01 { +v .0025 }
	}
	a 0; a1 .2; d 160
}

// Bell lead: FM with an inharmonic ratio and a long ring
MBell(P V=1 L=400)
{
	struct { fm2; panmix }
	pan -.15; set
	@p P; @p1 1.807; @a1 1.3; @a (V * .2)
	a1 .15; a (V * .09); d 150
	a (V * .04); d L
	a 0; d 400
}

MKick(V=1)
{
	struct { wtosc; panmix }
	w sine; @p -.8; @a (V * .75)
	p -2.6; d 70
	a 0; d 160
}

MSnare(V=1)
{
	struct {
		wtosc n; filter12 f; panmix pn 1 >
		wtosc s; panmix ps 1 >
	}
	n.w noise; @n.p 6; @f.cutoff 3; f.q .05; f.lp 0; f.bp .6; f.hp .4
	@n.a (V * .3)
	s.w sine; @s.p -.45; @s.a (V * .3)
	s.p -.8; s.a 0; d 60
	n.a 0; d 160
}

MHat(V=1 L=40)
{
	struct { wtosc; filter12; panmix }
	pan .25; set
	w noise; @p 7; @cutoff 5; q .05; lp 0; hp 1
	@a (V * .12)
	a 0; d L
}
'''

M = (0, 3, 7, 10)
J = (0, 4, 7, 11)

# Degree tokens used by bass and arpeggio patterns. Values index the
# chord's interval list (root, third, fifth, seventh) plus octaves.
DEG = {"r": (0, 0), "3": (1, 0), "5": (2, 0), "7": (3, 0),
       "R": (0, 12), "T": (1, 12), "F": (2, 12), "-5": (2, -12)}


def pitch(semi):
    return "%.4f" % (semi / 12.0)


def chord_semis(root, q):
    iv = M if q == "m" else J
    return root, iv


def norm(root, lo):
    """Put a chord root into the octave starting at semitone lo."""
    while root < lo:
        root += 12
    while root >= lo + 12:
        root -= 12
    return root


class Song:
    def __init__(self, name, bpm, chords, sections, bass, arp, drums,
                 melody, lead="MLead", delay=None, gain=1.0):
        self.name = name
        self.t = 60000.0 / bpm / 4.0	# ms per sixteenth
        self.chords = chords		# list of (root semitone, "m"/"M")
        self.sections = sections	# list of (layers set, repeat)
        self.bass = bass		# list of (step, degree, length steps)
        self.arp = arp			# 16 degree tokens or "." for rest
        self.drums = drums		# dict of style -> (kick, snare, hat)
        self.melody = melody		# list of (bar, step, semitone, len)
        self.lead = lead
        self.delay = delay or (380, 0.35, 285, 0.3, 427, 0.3)
        self.gain = gain


def bar_events(song, bar, layers, vol):
    """Return a list of (step, line) for one bar of the given layers."""
    root, q = song.chords[bar % len(song.chords)]
    iv = M if q == "m" else J
    ev = []
    t = song.t
    if "pad" in layers:
        pr = norm(root, -7)
        for i, semi in enumerate((pr, pr + iv[1], pr + iv[2], pr + 12)):
            ev.append((0, "MPad %s %.2f %d %.2f" % (
                pitch(semi), vol * 0.9, t * 16, (i - 1.5) * 0.3)))
    if "bass" in layers:
        br = norm(root, -27)
        for step, deg, ln in song.bass:
            k, o = DEG[deg]
            ev.append((step, "MBass %s %.2f %d" % (
                pitch(br + iv[k] + o), vol, t * ln)))
    if "arp" in layers:
        ar = norm(root, -5)
        for step, deg in enumerate(song.arp):
            if deg == ".":
                continue
            k, o = DEG[deg]
            x = -0.5 if step % 2 == 0 else 0.5
            ev.append((step, "MPluck %s %.2f %.1f" % (
                pitch(ar + iv[k] + o), vol, x)))
    for style in ("drive", "half", "hats"):
        if style in layers:
            kick, snare, hat = song.drums[style]
            for step in range(16):
                if kick[step] == "K":
                    ev.append((step, "MKick %.2f" % vol))
                if snare[step] == "S":
                    ev.append((step, "MSnare %.2f" % vol))
                if snare[step] == "s":
                    ev.append((step, "MSnare %.2f" % (vol * 0.4)))
                if hat[step] == "H":
                    ev.append((step, "MHat %.2f" % vol))
                if hat[step] == "h":
                    ev.append((step, "MHat %.2f" % (vol * 0.5)))
                if hat[step] == "O":
                    ev.append((step, "MHat %.2f 160" % vol))
    if "lead" in layers:
        mb = bar % 8
        for b, step, semi, ln in song.melody:
            if b == mb:
                ev.append((step, "%s %s %.2f %d" % (
                    song.lead, pitch(semi), vol, t * ln - 30)))
    return ev


def emit_section(song, layers, bars, vol, indent="\t\t"):
    out = []
    for bar in range(bars):
        ev = sorted(bar_events(song, bar, layers, vol), key=lambda e: e[0])
        now = 0
        for step, line in ev:
            if step > now:
                out.append("%sd %d" % (indent, (step - now) * song.t))
                now = step
            out.append(indent + line)
        if now < 16:
            out.append("%sd %d" % (indent, (16 - now) * song.t))
    return out


def emit_song(song):
    lines = ["", "", "  " + "/" * 75, " // " + song.name,
             "/" * 77, "", "%s()" % song.name, "{"]
    first = True
    for layers, reps, *loop in song.sections:
        if loop and loop[0] == "loop" and first:
            lines.append("\tfor {")
            first = False
        ind = "\t\t" if not first else "\t"
        for _ in range(reps):
            lines.append("%s// %s" % (ind, " ".join(sorted(layers))))
            lines.extend(emit_section(song, layers, 8, song.gain, ind))
    if not first:
        lines.append("\t}")
    lines.append("}")
    return lines


def emit_wrapper(export, song):
    fb, fg, ld, lg, rd, rg = song.delay
    return ["", "export %s()" % export, "{",
            "\tstruct {",
            "\t\tinline 0 2",
            "\t\tfbdelay 2 2",
            "\t\tpanmix 2 >",
            "\t}",
            "\tfbdelay %d; fbgain %.2f" % (fb, fg),
            "\tldelay %d; lgain %.2f" % (ld, lg),
            "\trdelay %d; rgain %.2f" % (rd, rg),
            "\tvol .45",
            "\tset",
            "\t1:%s" % song.name,
            "\tend",
            ".stop\tvol 0; d 1000",
            "\tkill 1",
            "\t1() {",
            "\t\tforce stop",
            "\t}",
            "}"]


# Drum styles shared by most songs
DRIVE = ("K.......K.K.....", "....S.......S..s", "H.h.H.h.H.h.H.hO")
HALF = ("K.........K.....", "........S.......", "H...h...H...h...")
HATS = ("................", "................", "..h...h...h...h.")
DR = {"drive": DRIVE, "half": HALF, "hats": HATS}

PAD = {"pad"}
INTRO = {"pad", "arp"}
GROOVE = {"pad", "arp", "bass", "half"}
MAIN = {"pad", "arp", "bass", "drive", "lead"}
LIFT = {"pad", "bass", "half", "lead"}
BREAK = {"pad", "arp", "hats"}

SONGS = {
    # Title: A minor, unhurried. Bell melody over pads, light percussion.
    "S_TITLESONG": Song(
        "HTitle", 84,
        [(9, "m"), (5, "M"), (0, "M"), (7, "M")] * 2,
        [(INTRO, 1), ({"pad", "arp", "bass", "lead"}, 1, "loop"),
         ({"pad", "arp", "bass", "half", "lead"}, 1),
         (BREAK, 1)],
        [(0, "r", 6), (6, "r", 2), (8, "5", 4), (12, "R", 4)],
        ["r", "5", "R", "T", "F", "T", "R", "5"] * 2,
        DR,
        [(0, 0, 16, 6), (0, 6, 14, 2), (0, 8, 12, 4), (0, 12, 11, 4),
         (1, 0, 9, 8), (1, 8, 12, 4), (1, 12, 17, 4),
         (2, 0, 16, 8), (2, 8, 19, 4), (2, 12, 16, 4),
         (3, 0, 14, 12), (3, 12, 11, 4),
         (4, 0, 12, 4), (4, 4, 16, 4), (4, 8, 21, 8),
         (5, 0, 19, 6), (5, 6, 17, 2), (5, 8, 16, 4), (5, 12, 12, 4),
         (6, 0, 14, 8), (6, 8, 16, 4), (6, 12, 19, 4),
         (7, 0, 14, 8), (7, 8, 11, 8)],
        lead="MBell", delay=(530, 0.4, 357, 0.35, 714, 0.35)),

    # Region 1: D minor, driving
    "S_INGAMESONG1": Song(
        "HRegion1", 118,
        [(2, "m"), (10, "M"), (0, "M"), (9, "m")] * 2,
        [(INTRO, 1), (GROOVE, 1, "loop"), (MAIN, 2), (BREAK, 1)],
        [(0, "r", 2), (2, "r", 2), (4, "R", 2), (6, "r", 2),
         (8, "r", 2), (10, "5", 2), (12, "R", 2), (14, "r", 2)],
        ["r", "5", "R", "5", "T", "5", "R", "5"] * 2,
        DR,
        [(0, 0, 9, 4), (0, 4, 14, 4), (0, 8, 17, 6), (0, 14, 16, 2),
         (1, 0, 14, 8), (1, 8, 17, 4), (1, 12, 14, 4),
         (2, 0, 16, 6), (2, 6, 19, 2), (2, 8, 16, 4), (2, 12, 12, 4),
         (3, 0, 16, 12),
         (4, 0, 17, 4), (4, 4, 16, 4), (4, 8, 14, 4), (4, 12, 9, 4),
         (5, 0, 10, 6), (5, 6, 14, 2), (5, 8, 17, 8),
         (6, 0, 19, 4), (6, 4, 17, 4), (6, 8, 16, 4), (6, 12, 19, 4),
         (7, 0, 21, 8), (7, 8, 16, 8)]),

    # Region 2: E phrygian, tense
    "S_INGAMESONG2": Song(
        "HRegion2", 126,
        [(4, "m"), (5, "M"), (4, "m"), (2, "M"),
         (0, "M"), (2, "M"), (4, "m"), (4, "m")],
        [(INTRO, 1), (GROOVE, 1, "loop"), (MAIN, 2), (LIFT, 1)],
        [(0, "r", 3), (3, "r", 3), (6, "r", 2), (8, "R", 2),
         (10, "r", 2), (12, "3", 2), (14, "5", 2)],
        ["r", "3", "5", "R", "5", "3", "r", "3"] * 2,
        DR,
        [(0, 0, 16, 3), (0, 3, 17, 3), (0, 6, 16, 2), (0, 8, 11, 8),
         (1, 0, 12, 4), (1, 4, 9, 4), (1, 8, 17, 4), (1, 12, 16, 4),
         (2, 0, 19, 6), (2, 6, 17, 2), (2, 8, 16, 8),
         (3, 0, 18, 4), (3, 4, 14, 4), (3, 8, 9, 8),
         (4, 0, 16, 4), (4, 4, 19, 4), (4, 8, 24, 8),
         (5, 0, 23, 4), (5, 4, 21, 4), (5, 8, 18, 8),
         (6, 0, 19, 3), (6, 3, 18, 3), (6, 6, 16, 2), (6, 8, 11, 4),
         (6, 12, 16, 4),
         (7, 0, 16, 12)]),

    # Region 3: G major, open and bright
    "S_INGAMESONG3": Song(
        "HRegion3", 110,
        [(7, "M"), (2, "M"), (4, "m"), (0, "M")] * 2,
        [(INTRO, 1), (GROOVE, 1, "loop"), (MAIN, 2), (BREAK, 1)],
        [(0, "r", 4), (4, "5", 2), (6, "r", 2), (8, "R", 4),
         (12, "5", 2), (14, "3", 2)],
        ["r", "3", "5", "3", "R", "5", "3", "5"] * 2,
        DR,
        [(0, 0, 11, 4), (0, 4, 14, 4), (0, 8, 19, 6), (0, 14, 18, 2),
         (1, 0, 18, 8), (1, 8, 21, 4), (1, 12, 18, 4),
         (2, 0, 19, 6), (2, 6, 16, 2), (2, 8, 11, 8),
         (3, 0, 12, 4), (3, 4, 16, 4), (3, 8, 19, 4), (3, 12, 16, 4),
         (4, 0, 14, 8), (4, 8, 19, 8),
         (5, 0, 21, 6), (5, 6, 19, 2), (5, 8, 18, 8),
         (6, 0, 16, 4), (6, 4, 19, 4), (6, 8, 23, 8),
         (7, 0, 21, 8), (7, 8, 19, 8)]),

    # Region 4: F sharp minor, mysterious
    "S_INGAMESONG4": Song(
        "HRegion4", 104,
        [(6, "m"), (2, "M"), (9, "M"), (4, "M")] * 2,
        [(INTRO, 1), (GROOVE, 1, "loop"), (MAIN, 2), (BREAK, 1)],
        [(0, "r", 6), (6, "r", 2), (8, "5", 4), (12, "3", 4)],
        ["r", "5", "T", "5", "R", "5", "T", "F"] * 2,
        DR,
        [(0, 0, 13, 8), (0, 8, 18, 8),
         (1, 0, 16, 4), (1, 4, 14, 4), (1, 8, 9, 8),
         (2, 0, 13, 6), (2, 6, 16, 2), (2, 8, 21, 8),
         (3, 0, 20, 8), (3, 8, 16, 4), (3, 12, 11, 4),
         (4, 0, 21, 4), (4, 4, 20, 4), (4, 8, 18, 8),
         (5, 0, 18, 4), (5, 4, 16, 4), (5, 8, 14, 8),
         (6, 0, 16, 6), (6, 6, 13, 2), (6, 8, 9, 8),
         (7, 0, 11, 8), (7, 8, 8, 8)],
        lead="MBell"),

    # Region 5: C minor, the last stretch
    "S_INGAMESONG5": Song(
        "HRegion5", 134,
        [(0, "m"), (8, "M"), (3, "M"), (10, "M")] * 2,
        [(INTRO, 1), (GROOVE, 1, "loop"), (MAIN, 2), (LIFT, 1)],
        [(0, "r", 2), (2, "R", 2), (4, "r", 2), (6, "R", 2),
         (8, "r", 2), (10, "R", 2), (12, "5", 2), (14, "R", 2)],
        ["r", "R", "5", "R", "T", "R", "5", "R"] * 2,
        DR,
        [(0, 0, 19, 3), (0, 3, 20, 3), (0, 6, 19, 2), (0, 8, 15, 4),
         (0, 12, 12, 4),
         (1, 0, 12, 4), (1, 4, 15, 4), (1, 8, 20, 8),
         (2, 0, 19, 6), (2, 6, 17, 2), (2, 8, 15, 4), (2, 12, 10, 4),
         (3, 0, 14, 8), (3, 8, 17, 8),
         (4, 0, 24, 4), (4, 4, 22, 4), (4, 8, 19, 8),
         (5, 0, 20, 4), (5, 4, 19, 4), (5, 8, 15, 8),
         (6, 0, 19, 4), (6, 4, 22, 4), (6, 8, 27, 8),
         (7, 0, 26, 8), (7, 8, 22, 8)]),

    # Epilogue: C major, the long way home
    "S_EPILOGUE": Song(
        "HEpilogue", 76,
        [(0, "M"), (7, "M"), (9, "m"), (5, "M"),
         (0, "M"), (5, "M"), (7, "M"), (0, "M")],
        [(PAD, 1), ({"pad", "arp", "lead"}, 1, "loop"),
         ({"pad", "arp", "bass", "lead"}, 1), (BREAK, 1)],
        [(0, "r", 8), (8, "5", 8)],
        ["r", "5", "R", "T", "F", "T", "R", "5"] * 2,
        DR,
        [(0, 0, 16, 8), (0, 8, 19, 8),
         (1, 0, 14, 8), (1, 8, 11, 8),
         (2, 0, 12, 4), (2, 4, 16, 4), (2, 8, 21, 8),
         (3, 0, 19, 8), (3, 8, 17, 8),
         (4, 0, 16, 6), (4, 6, 14, 2), (4, 8, 12, 8),
         (5, 0, 21, 8), (5, 8, 24, 8),
         (6, 0, 23, 8), (6, 8, 26, 8),
         (7, 0, 24, 16)],
        lead="MBell", delay=(610, 0.45, 410, 0.4, 820, 0.4)),
}


def main():
    lines = HEADER.splitlines() + INSTRUMENTS.splitlines()
    for export, song in SONGS.items():
        lines.extend(emit_song(song))
        lines.extend(emit_wrapper(export, song))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.normpath(OUT), len(lines), "lines")


if __name__ == "__main__":
    main()
