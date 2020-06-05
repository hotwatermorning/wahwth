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
        Frequency = 0,
        HighFreq,
        LowFreq,
        QFactor,
        NumParams,
    };
    
    juce::AudioParameterFloat *freq_        = nullptr;
    juce::AudioParameterFloat *low_freq_    = nullptr;
    juce::AudioParameterFloat *high_freq_   = nullptr;
    juce::AudioParameterFloat *qfactor_     = nullptr;
    
    // BPF のカットオフ周波数の位置パラメータの範囲
    // カットオフ周波数の位置パラメータが実際にどの周波数になるかは、 LowFreq/HighFreq のパラメータの状態による。
    static constexpr float kFrequencyMin = 0.0;
    static constexpr float kFrequencyMax = 1.0;
    static constexpr float kFrequencyDefault = 0.0;
    
    static constexpr float kQFactorMin = 0.1;
    static constexpr float kQFactorMax = 7.0;
    static constexpr float kQFactorDefault = 3.0;
    
    static constexpr float kLowFreqMin = 100;
    static constexpr float kLowFreqMax = 1000;
    static constexpr float kLowFreqDefault = 120;
    
    static constexpr float kHighFreqMin = 2000;
    static constexpr float kHighFreqMax = 7500;
    static constexpr float kHighFreqDefault = 6000;
    
    static constexpr int kCameraIndexMin = 0;
    static constexpr int kCameraIndexMax = 127;
    static constexpr int kFlipFlagDefault = true;
    
    struct EditorData {
        int camera_index_ = kCameraIndexMin;
        bool flip_ = kFlipFlagDefault;
    };
    
    //! ファイルから読み込んだエディタ情報
    std::atomic<EditorData> editor_data_;
    
    using EditorDataUpdateCallback = std::function<void()>;
    
    void SetEditorDataUpdateCallback(EditorDataUpdateCallback cb);
    
    static constexpr int kOrder = 8;
    static constexpr int kNumBins = 1 << kOrder;
    
    //! @pre hist.size() >= kBinSize
    void getSampleHistory(std::vector<float> &hist);
    
    float paramToFreq(float param_value) const;
    float freqToParam(float freq) const;
    
private:    
    //! 画面上のスペクトル表示のために、処理したサンプルを kNumBins だけ保存しておく。
    //! ステレオ波形はモノラルにミックスダウンしてしまうため、 左右で逆相になっているような波形はうまく扱えないことに注意。
    std::mutex mtx_sample_history_;
    std::vector<float> sample_history_;
    int sample_write_pos_ = 0;    
    std::vector<juce::IIRFilter> filters_;
    float last_freq_;
    float last_q_;
    float last_qfactor_gain_;
    juce::SmoothedValue<float> smooth_freq_;
    juce::SmoothedValue<float> smooth_q_;
    
    std::mutex editor_data_update_mutex_;
    //! ファイルから Editor データを読み込んだ直後に呼び出されるコールバック
    EditorDataUpdateCallback on_update_editor_data_;
private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
