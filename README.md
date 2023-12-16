# Wyoming SDL2

[Wyoming protocol](https://github.com/rhasspy/wyoming) server that uses [py-sdl2](https://github.com/py-sdl/py-sdl2) for audio input/output.

## Installation

Install system dependencies:

``` sh
sudo apt-get update
sudo apt-get install libsdl2-mixer-2.0-0
```

Install the Python dependencies:

``` sh
script/setup
```


## Mic Example

Run a server that streams audio from the default microphone:

``` sh
script/run \
  --uri 'tcp://127.0.0.1:10600' \
  --mode mic
```

Add `--devices` to see available microphones (use `--mic-device <DEVICE>`). See `--help` for more `--mic-*` options.

## Snd Example

Run a server that streams audio from the default microphone:

``` sh
script/run \
  --uri 'tcp://127.0.0.1:10601' \
  --mode snd
```

Add `--devices` to see available playback devices (use `--snd-device <DEVICE>`). See `--help` for more `--snd-*` options.
