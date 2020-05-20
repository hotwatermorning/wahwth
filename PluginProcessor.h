#pragma once

#include <array>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

//==============================================================================
class AudioPluginAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    
    //==============================================================================

    //Processor
    enum Parameters
    {
        Bypass = 0,
        Frequency,
        NumParams,
    };
    
    juce::AudioParameterBool *bypass_;
    juce::AudioParameterFloat *freq_;
    
    static constexpr float kFrequencyMin = 120.0;
    static constexpr float kFrequencyMax = 6000.0;
    
    std::array<float, NumParams> parameters_;

    std::vector<juce::IIRFilter> filters_;
    float last_freq_;
    juce::SmoothedValue<float> smooth_freq_;
    
    static constexpr int kOrder = 8;
    static constexpr int kNumBins = 1 << kOrder;
    
    //! @pre hist.size() >= kBinSize
    void getSampleHistory(std::vector<float> &hist);
    
private:
    //! 画面上のスペクトル表示のために、処理したサンプルを kNumBins だけ保存しておく。
    //! ステレオ波形はモノラルにミックスダウンしてしまうため、 左右で逆相になっているような波形はうまく扱えないことに注意。
    std::mutex mtx_sample_history_;
    std::vector<float> sample_history_;
    int sample_write_pos_ = 0;
    
private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
