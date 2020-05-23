#include "PluginProcessor.h"
#include "PluginEditor.h"

//! 0.0 .. 1.0 の値を kFrequencyMin .. kFrequencyMax の値に変換する。
float paramToFreq(float param_value)
{
    constexpr double min_ = AudioPluginAudioProcessor::kFrequencyMin;
    constexpr double max_ = AudioPluginAudioProcessor::kFrequencyMax;
    
    assert(0.0 <= param_value && param_value <= 1.0);
    
    return pow(max_ / min_, param_value) * min_;
}

float freqToParam(float freq)
{
    constexpr double min_ = AudioPluginAudioProcessor::kFrequencyMin;
    constexpr double max_ = AudioPluginAudioProcessor::kFrequencyMax;
    
    auto log_with_base = [](double base, double x) {
        // 底を base にして対数を計算する
        return log(x) / log(base);
    };

    double const param = log_with_base(max_ / min_, (freq / min_));
    
    assert(0 <= param && param <= 1.0);
}

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
{
    bypass_ = new AudioParameterBool("bypass", "Bypass", false,
                                     "",
                                     [](bool x, int max_len) -> juce::String { return "Bypass"; },
                                     [](juce::String const &str) -> bool { return str.compareIgnoreCase("Bypass") == 0; }
                                     );
    
    freq_ = new AudioParameterFloat("freq", "Frequency", { 0.0, 1.0 }, 0.0,
                                    "Hz", AudioProcessorParameter::genericParameter,
                                    [](float value, int max_len) -> juce::String { return juce::String(paramToFreq(value)); },
                                    [](juce::String const &str) -> float {
                                        try {
                                            return freqToParam(std::stof(str.toRawUTF8()));
                                        } catch(...) {
                                            return 0.0;
                                        }
                                    }
                                    );
    addParameter(bypass_);
    addParameter(freq_);
    sample_history_.reserve(kNumBins * 2);
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    juce::ignoreUnused (sampleRate, samplesPerBlock);
    
    last_freq_ = kFrequencyMin;
    smooth_freq_.setTargetValue(kFrequencyMin);
    smooth_freq_.reset(sampleRate, 4.0 / sampleRate);

    filters_.resize(getTotalNumOutputChannels());
    std::fill(sample_history_.begin(), sample_history_.end(), 0.0);
}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);

    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // In case we have more outputs than inputs, this code clears any output
    // channels that didn't contain input data, (because these aren't
    // guaranteed to be empty - they may contain garbage).
    // This is here to avoid people getting screaming feedback
    // when they first compile a plugin, but obviously you don't need to keep
    // this code if your algorithm always overwrites all the output channels.
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    bool const bypass = bypass_->get();
    float const freq = paramToFreq(freq_->get());
    smooth_freq_.setTargetValue(freq);
    float const smoothed_freq = smooth_freq_.getNextValue();
    if(freq != smoothed_freq) {
        last_freq_ = smoothed_freq;
        
        auto coeff = juce::IIRCoefficients::makeBandPass(getSampleRate(), smoothed_freq, 5.0);
        for(auto &f: filters_) {
            f.setCoefficients(coeff);
        }
    }
    
    if(bypass) {
        for(int channel = 0; channel < totalNumInputChannels; ++channel) {
            auto* data = buffer.getWritePointer (channel);
            std::fill_n(data, buffer.getNumSamples(), 0.0);
        }
    } else {
        for(int channel = 0; channel < totalNumInputChannels; ++channel) {
            auto* data = buffer.getWritePointer (channel);
            filters_[channel].processSamples(data, buffer.getNumSamples());
        }
    }
    
    {
        std::unique_lock lock(mtx_sample_history_);
    
        for(int i = 0, end = buffer.getNumSamples(); i < end; ++i) {
            sample_history_[(i + sample_write_pos_) % kNumBins] = 0;
        }
        
        for(int channel = 0; channel < totalNumInputChannels; ++channel) {
            auto const * const data = buffer.getReadPointer (channel);
            
            for(int i = 0, end = buffer.getNumSamples(); i < end; ++i) {
                sample_history_[(i + sample_write_pos_) % kNumBins] += data[i];
            }
        }
        
        sample_write_pos_ = (buffer.getNumSamples() + sample_write_pos_) % kNumBins;
    }
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);
}

void AudioPluginAudioProcessor::getSampleHistory(std::vector<float> &hist)
{
    assert(hist.size() >= kNumBins);
    std::unique_lock lock(mtx_sample_history_);
    
    for(int i = 0, end = kNumBins; i < end; ++i) {
        hist[i] = sample_history_[(i + sample_write_pos_) % kNumBins];
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}
