# What's this?

This is a Wah-Wah effector audio plugin which can be manipulated with your mouth.

![captured.gif](https://user-images.githubusercontent.com/359226/82833785-0ddfba00-9efa-11ea-8d9e-ca701dbfb370.gif)

## Prerequisites

* [CMake](https://cmake.org/) 3.17 or later
* Visual Studio 2019 or later
* Xcode 11.3

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

