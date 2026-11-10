/*
 * a2render - render an Audiality 2 program from an .a2s file to a WAV file
 *
 * Usage: a2render <file.a2s> <program> <seconds> <out.wav> [args...]
 *
 * Runs the exported program in an off-line Audiality 2 state and writes
 * 16 bit mono WAV at 48 kHz. Prints peak and RMS levels, so sound design
 * work can be checked for clipping and silence without listening.
 *
 * Copyright 2026 Andrew Ahlfield. GPL-2.0-or-later, like the game code.
 *
 * Build: cc -O2 -o a2render a2render.c $(pkg-config --cflags --libs audiality2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "audiality2.h"

#define	RATE	48000
#define	BUF	256

static void put32(FILE *f, uint32_t v)
{
	fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f);
	fputc((v >> 16) & 0xff, f); fputc((v >> 24) & 0xff, f);
}

static void put16(FILE *f, uint16_t v)
{
	fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f);
}

int main(int argc, const char *argv[])
{
	A2_driver *drv;
	A2_config *cfg;
	A2_interface *iface;
	A2_handle bank, prog, vh;
	int args[8];
	unsigned argcount = 0;
	unsigned length, frames = 0, i;
	double peak = 0.0, sum = 0.0;
	int16_t *out;
	FILE *f;

	if(argc < 5)
	{
		fprintf(stderr, "Usage: %s <file.a2s> <program> <seconds> "
				"<out.wav> [args...]\n", argv[0]);
		return 1;
	}
	for(i = 5; i < (unsigned)argc && argcount < 8; ++i)
		args[argcount++] = (int)(atof(argv[i]) * 65536.0);
	length = (unsigned)(atof(argv[3]) * RATE);
	if(!(out = calloc(length + BUF, sizeof(int16_t))))
		return 2;

	/* Off-line master state with a "buffer" driver; we pump it ourselves */
	if(!(drv = a2_NewDriver(A2_AUDIODRIVER, "buffer")))
		return 3;
	if(!(cfg = a2_OpenConfig(RATE, BUF, 1, A2_AUTOCLOSE)))
		return 3;
	if(a2_AddDriver(cfg, drv))
		return 3;
	if(!(iface = a2_Open(cfg)))
	{
		fprintf(stderr, "a2_Open(): %s\n",
				a2_ErrorString(a2_LastError()));
		return 4;
	}
	if((bank = a2_Load(iface, argv[1], 0)) < 0)
	{
		fprintf(stderr, "a2_Load(\"%s\"): %s\n", argv[1],
				a2_ErrorString(-bank));
		return 5;
	}
	if((prog = a2_Get(iface, bank, argv[2])) < 0)
	{
		fprintf(stderr, "a2_Get(\"%s\"): %s\n", argv[2],
				a2_ErrorString(-prog));
		return 6;
	}
	if((vh = a2_Starta(iface, a2_RootVoice(iface), prog, argcount,
			args)) < 0)
	{
		fprintf(stderr, "a2_Starta(): %s\n", a2_ErrorString(-vh));
		return 7;
	}
	while(frames < length)
	{
		int res;
		int32_t *buf = ((A2_audiodriver *)drv)->buffers[0];
		unsigned frag = BUF;
		if(frag > length - frames)
			frag = length - frames;
		if((res = a2_Run(iface, frag)) < 0)
		{
			fprintf(stderr, "a2_Run(): %s\n",
					a2_ErrorString(-res));
			return 8;
		}
		for(i = 0; i < frag; ++i)
		{
			/* 8:24 fixed point to 16 bit */
			double s = buf[i] / 16777216.0;
			if(fabs(s) > peak)
				peak = fabs(s);
			sum += s * s;
			if(s > 1.0)
				s = 1.0;
			else if(s < -1.0)
				s = -1.0;
			out[frames + i] = (int16_t)(s * 32767.0);
		}
		frames += frag;
	}
	if(a2_LastRTError(iface))
		fprintf(stderr, "Realtime error: %s\n",
				a2_ErrorString(a2_LastRTError(iface)));
	printf("%-22s %5.2f s  peak %.3f (%5.1f dBFS)  rms %5.1f dBFS\n",
			argv[2], (double)length / RATE, peak,
			20 * log10(peak + 1e-9),
			10 * log10(sum / (length ? length : 1) + 1e-12));

	if(!(f = fopen(argv[4], "wb")))
		return 9;
	fwrite("RIFF", 1, 4, f); put32(f, 36 + length * 2);
	fwrite("WAVEfmt ", 1, 8, f); put32(f, 16); put16(f, 1); put16(f, 1);
	put32(f, RATE); put32(f, RATE * 2); put16(f, 2); put16(f, 16);
	fwrite("data", 1, 4, f); put32(f, length * 2);
	for(i = 0; i < length; ++i)
		put16(f, (uint16_t)out[i]);
	fclose(f);
	a2_Close(iface);
	free(out);
	return 0;
}
