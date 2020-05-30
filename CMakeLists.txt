# Example Audio Plugin CMakeLists.txt

# To get started on a new plugin, copy this entire folder (containing this file and C++ sources) to
# a convenient location, and then start making modifications.

# The first line of any CMake project should be a call to `cmake_minimum_required`, which checks
# that the installed CMake will be able to understand the following CMakeLists, and ensures that
# CMake's behaviour is compatible with the named version. This is a standard CMake command, so more
# information can be found in the CMake docs.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(DLIB_NO_GUI_SUPPORT YES)

cmake_minimum_required(VERSION 3.15)

# The top-level CMakeLists.txt file for a project must contain a literal, direct call to the
# `project()` command. `project()` sets up some helpful variables that describe source/binary
# directories, and the current project version. This is a standard CMake command.

project(Wahwth_Plugin VERSION 0.0.1)

# If you've installed JUCE somehow (via a package manager, or directly using the CMake install
# target), you'll need to tell this project that it depends on the installed copy of JUCE. If you've
# included JUCE directly in your source tree (perhaps as a submodule), you'll need to tell CMake to
# include that subdirectory as part of the build.

# find_package(JUCE CONFIG REQUIRED)        # If you've installed JUCE to your system
# or
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/juce)                    # If you've put JUCE in a subdirectory called JUCE
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/dlib)

# If you are building a VST2 or AAX plugin, CMake needs to be told where to find these SDKs on your
# system. This setup should be done before calling `juce_add_plugin`.

# juce_set_vst2_sdk_path(...)
# juce_set_aax_sdk_path(...)

# `juce_add_plugin` adds a static library target with the name passed as the first argument
# (AudioPluginExample here). This target is a normal CMake target, but has a lot of extra properties set
# up by default. As well as this shared code static library, this function adds targets for each of
# the formats specified by the FORMATS arguments. This function accepts many optional arguments.
# Check the readme at `examples/CMake` in the JUCE repo for the full list.

juce_add_plugin(Wahwth
    VERSION 0.9.1                               # Set this if the plugin version is different to the project version
    # ICON_BIG ...                              # ICON_* arguments specify a path to an image file to use as an icon for the Standalone
    # ICON_SMALL ...
    COMPANY_NAME "diatonic.jp"                  # Specify the name of the plugin's author
    IS_SYNTH FALSE                              # Is this a synth or an effect?
    NEEDS_MIDI_INPUT FALSE               # Does the plugin need midi input?
    NEEDS_MIDI_OUTPUT FALSE              # Does the plugin need midi output?
    IS_MIDI_EFFECT FALSE                 # Is this plugin a MIDI effect?
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE    # Does the editor need keyboard focus?
    # COPY_PLUGIN_AFTER_BUILD TRUE/FALSE        # Should the plugin be installed to a default location after building?
    PLUGIN_MANUFACTURER_CODE Dtcj               # A four-character manufacturer id with at least one upper-case character
    PLUGIN_CODE Whwt                            # A unique four-character plugin id with at least one upper-case character
    FORMATS AU VST3 Standalone                  # The formats to build. Other valid formats are: AAX Unity VST AU AUv3
    CAMERA_PERMISSION_ENABLED TRUE
    CAMERA_PERMISSION_TEXT "Use camera to change effect parameter."
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "Use microphone to test audio effect."
    PRODUCT_NAME "Wahwth"        # The name of the final executable, which can differ from the target name
    )

# `juce_generate_juce_header` will create a JuceHeader.h for a given target, which will be generated
# into your build tree. This should be included with `#include <JuceHeader.h>`. The include path for
# this header will be automatically added to the target. The main function of the JuceHeader is to
# include all your JUCE module headers; if you're happy to include module headers directly, you
# probably don't need to call this.

juce_generate_juce_header(Wahwth)

# `target_sources` adds source files to a target. We pass the target that needs the sources as the
# first argument, then a visibility parameter for the sources (PRIVATE is normally best practice,
# although it doesn't really affect executable targets). Finally, we supply a list of source files
# that will be built into the target. This is a standard CMake command.

target_sources(Wahwth PRIVATE
    PluginEditor.cpp
    PluginProcessor.cpp
    PluginEditor.h
    PluginProcessor.h
    AboutDialog.cpp
    AboutDialog.h
    )

# `target_compile_definitions` adds some preprocessor definitions to our target. In a Projucer
# project, these might be passed in the 'Preprocessor Definitions' field. JUCE modules also make use
# of compile definitions to switch certain features on/off, so if there's a particular feature you
# need that's not on by default, check the module header for the correct flag to set here. These
# definitions will be visible both to your code, and also the JUCE module code, so for new
# definitions, pick unique names that are unlikely to collide! This is a standard CMake command.

target_compile_definitions(Wahwth
    PUBLIC
    # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
    JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_plugin` call
    JUCE_USE_CURL=0     # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_plugin` call
    JUCE_USE_CAMERA=1
    JUCE_VST3_CAN_REPLACE_VST2=0
    )

# If your target needs extra binary assets, you can add them here. The first argument is the name of
# a new static library target that will include all the binary resources. There is an optional
# `NAMESPACE` argument that can specify the namespace of the generated binary data class. Finally,
# the SOURCES argument should be followed by a list of source files that should be built into the
# static library. These source files can be of any kind (wav data, images, fonts, icons etc.).
# Conversion to binary-data will happen when your target is built.

# juce_add_binary_data(AudioPluginData SOURCES
#     ${CMAKE_CURRENT_SOURCE_DIR}/shape_predictor_68_face_landmarks.dat
#     )

# `target_link_libraries` links libraries and JUCE modules to other libraries or executables. Here,
# we're linking our executable target to the `juce::juce_audio_utils` module. Inter-module
# dependencies are resolved automatically, so `juce_core`, `juce_events` and so on will also be
# linked automatically. If we'd generated a binary data target above, we would need to link to it
# here too. This is a standard CMake command.

target_link_libraries(Wahwth PRIVATE
    # AudioPluginData           # If we'd created a binary data target, we'd link to it here
    juce::juce_audio_utils
    juce::juce_gui_extra
    juce::juce_video
    juce::juce_dsp
    dlib::dlib
    )

target_include_directories(Wahwth
    PUBLIC
    dlib::dlib
    )

target_compile_options(Wahwth
    PUBLIC
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Werror=return-type>
    $<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>
    )

set(LANDMARK_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shape_predictor_68_face_landmarks.dat")
set(BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/Wahwth_artefacts/$<CONFIG>")

if(MSVC)
    # copy command can't treat slash as a path separator unlike cd command.
    string(REGEX REPLACE "/" "\\\\" LANDMARK_FILE "${LANDMARK_FILE}")
    string(REGEX REPLACE "/" "\\\\" BINARY_DIR "${BINARY_DIR}")

    add_custom_command(
        TARGET Wahwth
        POST_BUILD
        COMMAND echo mkdir for standalone
        COMMAND mkdir "\"${BINARY_DIR}\\Standalone\"" & cmd /c exit 0
        COMMAND echo mkdir for VST3
        COMMAND mkdir "\"${BINARY_DIR}\\VST3\\Wahwth.vst3\\Contents\\Resources\"" & cmd /c exit 0
        COMMAND echo copy the landmark file for standalone
        COMMAND copy "\"${LANDMARK_FILE}\"" "\"${BINARY_DIR}\\Standalone\\\""
        COMMAND echo copy the landmark file for VST3
        COMMAND copy "\"${LANDMARK_FILE}\"" "\"${BINARY_DIR}\\VST3\\Wahwth.vst3\\Contents\\Resources\\\""
        COMMENT "Deploy the landmark file."
        )
else()
    add_custom_command(
        TARGET Wahwth
        POST_BUILD
        COMMAND echo mkdir for standalone
        COMMAND mkdir -p "\"${BINARY_DIR}/Standalone\""
        COMMAND echo mkdir for VST3
        COMMAND mkdir -p "\"${BINARY_DIR}/VST3/Wahwth.vst3/Contents/Resources/\""
        COMMAND echo mkdir for AU
        COMMAND mkdir -p "\"${BINARY_DIR}/AU/Wahwth.component/Contents/Resources/\""
        COMMAND echo copy the landmarkfile for standalone
        COMMAND cp "\"${LANDMARK_FILE}\"" "\"${BINARY_DIR}/Standalone\""
        COMMAND echo copy the landmarkfile for VST3
        COMMAND cp "\"${LANDMARK_FILE}\"" "\"${BINARY_DIR}/VST3/Wahwth.vst3/Contents/Resources/\""
        COMMAND echo copy the landmarkfile for AU
        COMMAND cp "\"${LANDMARK_FILE}\"" "\"${BINARY_DIR}/AU/Wahwth.component/Contents/Resources/\""
        COMMENT "Deploy the landmark file."
        )
endif()
