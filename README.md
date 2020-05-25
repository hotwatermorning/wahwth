# What's this?

This is a Wah-Wah effector plugin which can be manipulated with your mouth.

## Requirements

* OpenCV (4.3 or later)
* dlib (693aa0a7 or later)

## How to build

```cpp
cd /path/to/wahwth
git submodule update --init --recursive
mkdir build
cd build
cmake -GXcode ..
open Wahwth_Plugin.xcodeproj
```

## License

This project is licensed under MIT License.

