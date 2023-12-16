#!/usr/bin/env python3
import argparse
import asyncio
import ctypes
import logging
import queue
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, Final, Optional

import sdl2
import sdl2.sdlmixer as mixer
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger()
_DIR = Path(__file__).parent

_MIC_QUEUE: "queue.Queue[Optional[bytes]]" = queue.Queue()
_MAX_MIC_QUEUE_SIZE = 10

_SND_QUEUE: "queue.Queue[bytes]" = queue.Queue()
_MAX_SND_QUEUE_SIZE = 100

MODE_MIC: Final = "mic"
MODE_SND: Final = "snd"


@dataclass
class State:
    is_running: bool = True
    mic_queues: Dict[str, "asyncio.Queue[Optional[bytes]]"] = field(
        default_factory=dict
    )
    mic_queues_lock: Lock = field(default_factory=Lock)


async def main() -> None:
    """Main entry point."""
    sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)
    if "--devices" in sys.argv[1:]:
        # List capture/playback devices
        num_capture = sdl2.SDL_GetNumAudioDevices(1)
        for i in range(num_capture):
            device_name = sdl2.SDL_GetAudioDeviceName(i, 1).decode("utf-8")
            print("mic", i, f'"{device_name}"')

        print("---")
        num_playback = sdl2.SDL_GetNumAudioDevices(0)
        for i in range(num_playback):
            device_name = sdl2.SDL_GetAudioDeviceName(i, 0).decode("utf-8")
            print("snd", i, f'"{device_name}"')

        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--mode",
        required=True,
        choices=(MODE_MIC, MODE_SND),
        help="Microphone input or sound output mode",
    )
    #
    parser.add_argument(
        "--mic-device", help="Name of microphone device to use (see --devices)"
    )
    parser.add_argument(
        "--mic-rate",
        type=int,
        default=16000,
        help="Sample rate of microphone audio (hertz)",
    )
    parser.add_argument(
        "--mic-width",
        type=int,
        default=2,
        help="Sample width of microphone audio (bytes)",
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=1,
        help="Number of channels in microphone audio",
    )
    parser.add_argument(
        "--mic-samples-per-chunk",
        type=int,
        default=1024,
        help="Number of samples to read at a time from microphone",
    )
    #
    parser.add_argument(
        "--snd-device", help="Name of sound device to use (see --devices)"
    )
    parser.add_argument(
        "--snd-rate",
        type=int,
        default=22050,
        help="Sample rate of output audio (hertz)",
    )
    parser.add_argument(
        "--snd-width",
        default=2,
        help="Sample width of output audio (bytes)",
    )
    parser.add_argument(
        "--snd-channels",
        type=int,
        default=1,
        help="Number of channels in output audio",
    )
    parser.add_argument(
        "--snd-samples-per-chunk",
        type=int,
        default=1024,
        help="Number of samples to write to output audio device",
    )
    #
    parser.add_argument(
        "--devices", action="store_true", help="Print available devices and exit"
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    mic_id: Optional[int] = None
    if args.mode == MODE_MIC:
        # Microphone
        _LOGGER.debug("Initializing microphone")
        if args.mic_width == 2:
            mic_format = sdl2.AUDIO_S16SYS
        elif args.mic_width == 4:
            mic_format = sdl2.AUDIO_S32SYS
        else:
            raise ValueError("Only 2 and 4 byte mic widths are supported")

        mic_spec = sdl2.SDL_AudioSpec(
            args.mic_rate,
            mic_format,
            args.mic_channels,
            args.mic_samples_per_chunk,
            callback=mic_callback,
        )
        mic_id = sdl2.SDL_OpenAudioDevice(
            args.mic_device.encode("utf-8") if args.mic_device else None,
            1,
            mic_spec,
            None,
            0,
        )
        sdl2.SDL_PauseAudioDevice(mic_id, 0)
    elif args.mode == MODE_SND:
        # Sound
        _LOGGER.debug("Initializing output audio")

        if args.snd_width == 2:
            snd_format = sdl2.AUDIO_S16SYS
        elif args.snd_width == 4:
            snd_format = sdl2.AUDIO_S32SYS
        else:
            raise ValueError("Only 2 and 4 byte snd widths are supported")

        mixer.Mix_Init(0)
        mixer.Mix_OpenAudioDevice(
            args.snd_rate,
            snd_format,
            args.snd_channels,
            args.snd_samples_per_chunk,
            args.snd_device.encode("utf-8") if args.snd_device else None,
            0,
        )

    _LOGGER.info("Ready")

    state = State()
    loop = asyncio.get_running_loop()

    if args.mode == MODE_MIC:
        mic_distribute_thread = Thread(
            target=mic_distribute_proc, args=(state, loop), daemon=True
        )
        mic_distribute_thread.start()
    elif args.mode == MODE_SND:
        mixer.Mix_HookMusic(snd_music_hook, None)

    # Start server
    server = AsyncServer.from_uri(args.uri)

    try:
        if args.mode == MODE_MIC:
            await server.run(partial(MicEventHandler, args, state))
        elif args.mode == MODE_SND:
            await server.run(partial(SndEventHandler, args))
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Stopping")
        state.is_running = False

        if args.mode == MODE_MIC:
            assert mic_id is not None

            # Close microphone
            sdl2.SDL_PauseAudioDevice(mic_id, 1)
            sdl2.SDL_CloseAudioDevice(mic_id)

            # Stop thread
            _MIC_QUEUE.put_nowait(None)
            mic_distribute_thread.join()
        elif args.mode == MODE_SND:
            mixer.Mix_CloseAudio()

        _LOGGER.info("Stopped")


# -----------------------------------------------------------------------------

AudioSpecCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int
)


@AudioSpecCallback
def mic_callback(userdata, stream, stream_length) -> None:
    try:
        audio_bytes = ctypes.string_at(stream, stream_length)
        _MIC_QUEUE.put_nowait(audio_bytes)

        # Trim queue
        while _MIC_QUEUE.qsize() > _MAX_MIC_QUEUE_SIZE:
            _MIC_QUEUE.get()
    except Exception:
        _LOGGER.exception("Unexpected error in mic callback")


def mic_distribute_proc(state: State, loop: asyncio.AbstractEventLoop) -> None:
    try:
        while state.is_running:
            audio_bytes = _MIC_QUEUE.get()
            if audio_bytes is None:
                # Stop signal
                break

            with state.mic_queues_lock:
                for mic_queue in state.mic_queues.values():
                    loop.call_soon_threadsafe(mic_queue.put_nowait, audio_bytes)
    except Exception:
        _LOGGER.exception("Unexpected error in mic_distribute_proc")

        # Crash in the hope that we'll be restarted
        sys.exit(1)


class MicEventHandler(AsyncEventHandler):
    """Event handler for microphone clients."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        state: State,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.client_id = str(time.monotonic_ns())
        self.state = state
        self.mic_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self.run_task = asyncio.create_task(self.run_mic())
        self.is_running = True
        self.rate = self.cli_args.mic_rate
        self.width = self.cli_args.mic_width
        self.channels = self.cli_args.mic_channels

        _LOGGER.debug("Client connected: %s", self.client_id)
        with self.state.mic_queues_lock:
            self.state.mic_queues[self.client_id] = self.mic_queue

    async def handle_event(self, event: Event) -> bool:
        # Write only
        return True

    async def run_mic(self) -> None:
        try:
            await self.write_event(
                AudioStart(
                    rate=self.rate,
                    width=self.width,
                    channels=self.channels,
                ).event()
            )

            while self.is_running:
                audio_bytes = await self.mic_queue.get()
                if audio_bytes is None:
                    # Stop signal
                    await self.write_event(AudioStop().event())
                    break

                chunk = AudioChunk(
                    rate=self.rate,
                    width=self.width,
                    channels=self.channels,
                    audio=audio_bytes,
                    timestamp=time.monotonic_ns(),
                )
                await self.write_event(chunk.event())

                # Trim queue
                while self.mic_queue.qsize() > _MAX_MIC_QUEUE_SIZE:
                    await self.mic_queue.get()
        except Exception:
            _LOGGER.exception("Unexpected error in run_mic")

    async def disconnect(self) -> None:
        with self.state.mic_queues_lock:
            self.state.mic_queues.pop(self.client_id, None)

        # Stop signal to task
        self.is_running = False
        self.mic_queue.put_nowait(None)

        _LOGGER.debug("Client disconnected: %s", self.client_id)


# -----------------------------------------------------------------------------

HookMusicFunc = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int
)


@HookMusicFunc
def snd_music_hook(udata, stream, length) -> None:
    try:
        audio_bytes = _SND_QUEUE.get_nowait()
        ctypes.memmove(stream, audio_bytes, length)
    except queue.Empty:
        # Silence
        ctypes.memset(stream, 0, length)
    except Exception:
        _LOGGER.exception("Unexpected error in snd callback")


class SndEventHandler(AsyncEventHandler):
    """Event handler for sound clients."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.client_id = str(time.monotonic_ns())

        _LOGGER.debug("Client connected: %s", self.client_id)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            _SND_QUEUE.put_nowait(chunk.audio)

            while _SND_QUEUE.qsize() > _MAX_SND_QUEUE_SIZE:
                _SND_QUEUE.get()

        return True


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
