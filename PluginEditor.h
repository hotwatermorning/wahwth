#pragma once

#include <JuceHeader.h>

#include "PluginProcessor.h"

//==============================================================================
class AudioPluginAudioProcessorEditor
:   public juce::AudioProcessorEditor
,   juce::Slider::Listener
,   juce::ComboBox::Listener
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
    
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    
    void OnUpdateMouthAspectRatio();

    void sliderValueChanged(juce::Slider *s) override;
    void comboBoxChanged(juce::ComboBox *c) override;
    void mouseUp (juce::MouseEvent const&e) override;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
