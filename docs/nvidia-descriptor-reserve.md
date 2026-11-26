# Experimental capped descriptor reserve

On a device that cannot create the padded shader-visible view heap, the
host allocates at most 1,000,000 descriptors, with the memory-by-address
slots at the end. The same immutable process reserve determines the
mirrored prefix, dynamic range, reserved null slot, and shader clamp.
The shader cache already hashes the clamp; the PSO library hashes DXIL.

`VICTORY_DYNAMIC_SLOTS` is optional. Default: 65536 (unchanged).
Values are clamped to [65536, (1000000 - memory::SLOTS) / 2].
One slot in the reserve is permanently null. A larger reserve reduces
the directly mirrored title slots, so high bindless indices may read
null even as more root tables fit. This is not a validated gameplay preset.

Inspired by trustno1-coder/VIctory-NVIDIA a9632bc8 (GPL-3.0).
The source's request of 524288 is also constrained by its heap layout.
This implementation keeps our preflight validation and refuses releases
that would free the null slot. It also refuses a fallback heap whose clamp
would conflict with already translated shaders.

Validate using CPU layout regression and isolated CI first. Compare
descriptor failures, directly mirrored slots, image correctness and
native frame times before changing the default. Do not change the reserve
mid-process. No game assets or launcher modifications.
