
# MERT Vamp Plugin

This is an **experimental** [Vamp plugin](https://vamp-plugins.org)
implementation of [MERT audio
features](https://github.com/yizhilll/MERT) corresponding to the
published pre-trained
[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M).

## What is it?

MERT is a machine-learning model that converts music audio into
sequences of feature vectors that can serve as input for various
"music understanding" tasks. The features are not expected to be
directly interpretable on their own, but to capture patterns that can
be used in training further models for specific activities. See [MERT:
Acoustic Music Understanding Model with Large-Scale Self-supervised
Training](https://arxiv.org/abs/2306.00107) for more details.

This code implements the smallest current pre-trained MERT model at
the time of writing, as a Vamp plugin. That is, a native-code binary
that can be run in batch tools such as [Sonic
Annotator](https://vamp-plugins.org/sonic-annotator/) or graphical
ones such as [Sonic Visualiser](https://www.sonicvisualiser.org/), or
potentially adapted into other applications.

This code is **experimental** in the sense that it is a draft
implementation developed by comparing outputs against the original
PyTorch pre-trained model, that has not yet been put to use for any
serious purpose and is known to have some limitations. The code itself
is intended to be comprehensible and reusable and potentially handy
for other models.

## Compiling the plugin

Pre-compiled plugin binaries are available from the Github Releases
tab. To build your own:

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

The build can also detect and use other BLAS/CBLAS implementations
besides MKL or Accelerate, but by default it won't do so unless you
specify `-Duse_cblas`, because performance depends so much on the
specific BLAS implementation and is often slower than the default
code.

The build can also configure itself with or without thread support
using OpenMP or Apple's Dispatch library, depending on availability;
this is normally automatic.

All of these build configurations should produce identical results,
just at different speeds. Here's a comparative table of runtimes to
process a 20s test file using 8s chunks on an Intel i5-1340P (on CPU
only) running Arch Linux:

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

Chunking is handled in a completely naive way: the selected duration
is rounded to ensure an exact feature count, the input audio is split
into chunks at precisely the length needed for the rounded duration,
and the features are stuck together again without modification or
interpolation afterwards. Discontinuities can therefore occur at chunk
boundaries - an obvious area for improvement.

The plugin has 14 outputs, all returning feature vectors with 768
values.

The first output ("Convolutional embedding") is the output of the
convolutional preprocessor as supplied to the first attention layer in
the model.

The next 12 outputs ("Hidden layer N state" for N in 1-12) are
features extracted from the subsequent 12 rounds of attention layers,
with layer 12 being the final output from the model.

Finally there is an output reporting the mean values of each feature
bin, for each of the other outputs separately, across the whole input
duration. The features returned by this output (13 of them) are
timestamped so that a display in Sonic Visualiser or similar will show
them as a grid with the x-coordinate being output number; the
timestamps for this output otherwise have no meaning.

## Credits and copyright

The plugin was written by Chris Cannam in the Centre for Digital
Music, Queen Mary University of London, based on the MERT Python code
by Li et al. Any mistakes in the adaptation are totally my own
fault. See the [MERT
documentation](https://huggingface.co/m-a-p/MERT-v1-95M) for details
of the model, full credits, and accompanying citations.

Copyright (c) 2025 Queen Mary, University of London.

The plugin code is published under an MIT/X11 licence and the model
weights (included in the built artifact) are under Creative Commons
CC-BY-NC-4.0. The resulting plugin is therefore redistributable but
not for commercial use (and so not technically Open Source).

