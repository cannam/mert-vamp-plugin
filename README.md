
# MERT Vamp Plugin

This is an **experimental** [Vamp plugin](https://vamp-plugins.org)
implementation of [MERT audio
features](https://github.com/yizhilll/MERT) corresponding to the
published pre-trained
[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M).

## Compiling the plugin

```
$ ./repoint install
$ meson setup build
$ ninja -C build
```

The resulting binary will perform OK on a Mac (using Accelerate) but
be appallingly slow anywhere else. To speed it up, build using
libtorch, or at least Intel MKL for matrix acceleration. To explain:

This repo actually contains two different adaptations of the
model. One is a close conversion of the PyTorch code using the
libtorch C++ API; it can be found in the `cpp-libtorch`
subdirectory. The other is a naive C++ implementation that can be
compiled using MKL or Accelerate or without any external dependencies;
this is in `cpp-selfcontained`. The libtorch build can be much faster
if you have the framework properly configured, but that can be a bit
troublesome. The default, as above, will get you the self-contained
version without any extra performance libraries (except on the Mac
where they are system libraries).

To configure with libtorch, you need to tell the build where to look
for your libtorch install. If it is a system-wide installation,
something like this may be enough:

```
$ ./repoint install
$ meson setup build -Dlibtorch_path=/usr
$ ninja -C build
```

Similarly, if you lack libtorch but want to use the MKL to speed
things up a bit, something like

```
$ ./repoint install
$ meson setup build -Dmkl_path=/opt/intel/oneapi/mkl/latest
$ ninja -C build
```

The build can also configure itself with or without thread support
using OpenMP or Apple's Dispatch library, depending on availability;
this is normally automatic.

All of these build configurations should produce identical results,
just at different speeds. Here's a comparative table of runtimes to
process a 20s test file using 8s chunks on an Intel i5-1340P running
Arch Linux:

<table>
<tr><td><b>Configuration</b></td><td><b>Run time</b></td><td><b>Where is the extra time spent?</b></td></tr>
<tr><td>With libtorch</td><td>2.6 sec</td><td>n/a</td></tr>
<tr><td>Without libtorch, with Intel MKL and OpenMP</td><td>5.3 sec</td><td>Convolution</td></tr>
<tr><td>Without libtorch or Intel MKL, with OpenMP</td><td>13.2 sec</td><td>Matrix multiplication</td></tr>
<tr><td>Without libtorch, Intel MKL, or OpenMP</td><td>66.8 sec</td><td>Serial execution of both</td></tr>
</table>

## Parameters and Outputs

The plugin has only one adjustable parameter, "Chunk Duration"
(`chunk`). This controls the length in seconds of the chunks into
which the input audio will be split in order to feed them to the
model. Longer chunks may lead to the audio being processed more
quickly overall, but too long risks running out of memory, or timing
out on individual process calls, if the host uses a timeout.



## Credits and copyright

The plugin was written by Chris Cannam in the Centre for Digital
Music, Queen Mary University of London, based on the MERT Python code
by Li et al. Any mistakes in the adaptation are totally my own
fault. See the [MERT
documentation](https://huggingface.co/m-a-p/MERT-v1-95M) for details
of the model and accompanying citations.

Copyright (c) 2025 Queen Mary, University of London.

The plugin code is published under an MIT/X11 licence and the model
weights (included in the built artifact) are under Creative Commons
CC-BY-NC-4.0. The resulting plugin is therefore redistributable but
not for commercial use (and so not technically Open Source).

