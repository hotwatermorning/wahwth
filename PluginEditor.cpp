#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "AboutDialog.h"
#include <algorithm>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>

#define WAHWTH_USE_IMAGE_PROCESSING_THREAD 1

bool kShowFace = true;

int const kMinWidth = 400;
int const kMinHeight = 300;
int const kMaxWidth = 1200;
int const kMaxHeight = 900;
int const kDefaultWidth = 600;
int const kDefaultHeight = 450;
int const kProcessImageWidth = kDefaultWidth / 2;
int const kProcessImageHeight = kDefaultHeight / 2;

template<class PixelType>
struct dlib_pixel_type_to_juce_pixel_format;

template<>
struct dlib_pixel_type_to_juce_pixel_format<dlib::rgb_pixel>
{
    static constexpr juce::Image::PixelFormat pixel_format
    = juce::Image::PixelFormat::RGB;
};

template<>
struct dlib_pixel_type_to_juce_pixel_format<dlib::rgb_alpha_pixel>
{
    static constexpr juce::Image::PixelFormat pixel_format
    = juce::Image::PixelFormat::ARGB;
};

template<class PixelType>
struct JuceImageArray2d
{
    using pixel_type = PixelType;
    
    JuceImageArray2d(juce::Image img)
    {
        assert(img.isValid());
        assert(img.getPixelData()->pixelFormat
               == dlib_pixel_type_to_juce_pixel_format<PixelType>::pixel_format);
        img_ = img;
    }
    
    void swap(JuceImageArray2d &rhs)
    {
        std::swap(img_, rhs.img_);
    }
    
public:
    juce::Image img_;
};

template<class PixelType>
long        num_rows      (const JuceImageArray2d<PixelType>& img)
{
    return img.img_.getHeight();
}

template<class PixelType>
long        num_columns   (const JuceImageArray2d<PixelType>& img)
{
    return img.img_.getWidth();
}

// not supported.
// void        set_image_size(      JuceImageArray2d& img, long rows, long cols);

template<class PixelType>
void*       image_data    (      JuceImageArray2d<PixelType>& img)
{
    auto bmp = juce::Image::BitmapData(img.img_, juce::Image::BitmapData::readWrite);
    return bmp.data;
}

template<class PixelType>
const void* image_data    (const JuceImageArray2d<PixelType>& img)
{
    auto bmp = juce::Image::BitmapData(img.img_, juce::Image::BitmapData::readOnly);
    return bmp.data;
}

template<class PixelType>
long        width_step    (const JuceImageArray2d<PixelType>& img)
{
    auto bmp = juce::Image::BitmapData(img.img_, juce::Image::BitmapData::readOnly);
    return bmp.lineStride;
}

template<class PixelType>
void        swap          (      JuceImageArray2d<PixelType>& a, JuceImageArray2d<PixelType>& b)
{
    a.swap(b);
}

namespace dlib
{
    template <class PixelType>
    struct image_traits<JuceImageArray2d<PixelType>>
    {
        typedef typename JuceImageArray2d<PixelType>::pixel_type pixel_type;
    };
}

template<class T = void>
struct default_div
{
    T operator()(T const &t, double x) const {
        assert(x != 0);
        return t / x;
    }
};

template<>
struct default_div<void>
{
    template<class T>
    T operator()(T const &t, double x) const {
        assert(x != 0);
        return t / x;
    }
};

template<class T, class PlusFunc = std::plus<T>, class DivFunc = default_div<T>>
struct MovingAverageValue
{
    MovingAverageValue()
    :   plus_(PlusFunc{})
    ,   div_(DivFunc{})
    {}
    
    MovingAverageValue(PlusFunc plus, DivFunc div)
    :   plus_(plus)
    ,   div_(div)
    {}
    
    int GetNumHistory() const
    {
        return history_.size();
    }
    
    void SetNumHistory(int n, T const &init = T{})
    {
        history_.resize(n, init);
    }
    
    template<class U>
    void Push(U &&x)
    {
        history_[write_pos_] = std::move(x);
        write_pos_ = (write_pos_ + 1) % history_.size();
    }
    
    T GetAverage() const
    {
        T tmp = T{};
        
        auto const num = history_.size();
        for(int i = 0; i < num; ++i) {
            tmp = plus_(tmp, history_[(i + write_pos_) % num]);
        }
        
        return div_(tmp, (double)num);
    }
    
private:
    std::vector<T> history_;
    int write_pos_ = 0;
    PlusFunc plus_;
    DivFunc div_;
};

struct AudioPluginAudioProcessorEditor::Impl
:   public juce::CameraDevice::Listener
,   private juce::Thread
{
    using WF = juce::dsp::WindowingFunction<float>;
    using AP = AudioPluginAudioProcessor;
 
    static constexpr int kMouthPointsBegin = 48;
    static constexpr int kMouthPointsEnd = 68;
    static constexpr int kNumMouthPoints = kMouthPointsEnd - kMouthPointsBegin;
    static constexpr int kNumHistory = 3;
    
    Impl(AudioPluginAudioProcessorEditor *owner)
    :   juce::Thread("Image Processing Thread")
    ,   owner_(owner)
    {
        fft_ = std::make_unique<juce::dsp::FFT>(AP::kOrder);
        fft_buffer_.resize(AP::kNumBins * 2);
        wnd_ = std::make_unique<WF>(AP::kNumBins, WF::blackman);
        
        mtx_call_async_ = std::make_shared<std::mutex>();
        face_data_.SetNumHistory(kNumHistory);
    }
    
    ~Impl()
    {
        stopImageProcessingThread();
        closeCamera();
    }
    
    void startImageProcessingThread()
    {
        juce::Thread::startThread();
    }
    
    void stopImageProcessingThread()
    {
        if(isThreadRunning() == false) { return; }
        
        auto p = mtx_call_async_;
        {
            assert(p);
            // call_async 処理中は、その終了を待機する。
            std::unique_lock lock(*p);
            
            mtx_call_async_.reset();
        }
        
        signalThreadShouldExit();
        juce::Thread::stopThread(5000);
    }
    
    struct FaceData
    {
        using MouthPointArray = std::array<MovingAverageValue<dlib::dpoint>, kNumMouthPoints>;
        juce::Image image_;             //!< エディター画面で表示するカメラ画像
        MouthPointArray mouth_points_;  //!< エディター画面で表示する口の輪郭
        MovingAverageValue<double> mar_;
        int num_skipped_ = 0;           //!< 顔検出できなかったフレーム数。（検出できた時点で 0 にリセットされる）
        int num_processed_ = 0;
        
        void SetNumHistory(int n) {
            for(auto &p: mouth_points_) { p.SetNumHistory(n); }
            mar_.SetNumHistory(n);
        }
    };
    
    FaceData getFaceData() const
    {
        std::unique_lock lock(gui_mtx_);
        return face_data_;
    }
    
    //! 対数振幅スペクトルを取得
    //! s には、 [0..fs/2] の範囲の最新の対数振幅スペクトルのデータが書き込まれる。
    using SpectrumType = std::array<float, AP::kNumBins/2+1>;
    
    SpectrumType const & calculateLogAmplitudeSpectrum()
    {
        std::fill(fft_buffer_.begin(), fft_buffer_.end(), 0.0);
        owner_->processorRef.getSampleHistory(fft_buffer_);
        
        wnd_->multiplyWithWindowingTable(fft_buffer_.data(), AP::kNumBins);
        
        fft_->performRealOnlyForwardTransform(fft_buffer_.data());
        
        int const N = AP::kNumBins;
        assert(N == fft_->getSize());
        
        //! 直流成分とナイキスト周波数成分を N で正規化する。
        fft_buffer_[0] /= N;
        fft_buffer_[AP::kNumBins] /= N;
        
        //! それ以外の交流成分は、 N で正規化後、 複素共役の分を考慮して 2 倍する。
        std::transform(fft_buffer_.data() + 1,
                       fft_buffer_.data() + N,
                       fft_buffer_.data() + 1,
                       [&](float f) -> float { return f / (double)N * 2.0; });
        
        //! 対数振幅スペクトルを計算
        for(int i = 0, end = spectrum_.size(); i < end ; ++i) {
            auto amp_spec = std::sqrt(fft_buffer_[i*2] * fft_buffer_[i*2] +
                                      fft_buffer_[i*2+1] * fft_buffer_[i*2+1]);
            
            if(amp_spec == 0) {
                spectrum_[i] = -120;
            } else {
                spectrum_[i] = std::max<float>(-120, std::log10(amp_spec) * 20);
            }
        }
        
        return spectrum_;
    }
    
    bool openCamera(int index)
    {
        closeCamera();
        
        // This function crashes on Bitwig Studio 3.1.
        // Is there any workaround?
        cam_.reset(juce::CameraDevice::openDevice(index, kMinWidth, kMinHeight, kMaxWidth, kMaxHeight));
        if(cam_) {
            cam_->addListener(this);
            
            // このウィンドウは表示しない。
            // （画面が顔の動きと反転しているのでこの画像はそのまま使わない。代わりに画像処理スレッドで処理した画像を反転して表示する）
            viewer_ = cam_->createViewerComponent();
            return true;
        } else {
            return false;
        }
    }
    
    void closeCamera()
    {
        if(cam_) {
            delete viewer_;
            viewer_ = nullptr;
            cam_->removeListener(this);
            cam_.reset();
        }
    }
public:
    AudioPluginAudioProcessorEditor *owner_ = nullptr;
    std::unique_ptr<juce::CameraDevice> cam_;
    juce::Component *viewer_ = nullptr;
    
    juce::ComboBox cmb_camera_list_;
    juce::Slider sl_low_freq_;
    juce::Slider sl_high_freq_;
    juce::Slider sl_qfactor_;

    juce::Label lbl_low_freq_;
    juce::Label lbl_high_freq_;
    juce::Label lbl_qfactor_;
    
private:
    std::mutex mutable gui_mtx_;    //!< GUI スレッドと画像処理スレッドの排他制御
        
    std::vector<float> fft_buffer_;
    std::unique_ptr<juce::dsp::FFT> fft_;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> wnd_;
    
    FaceData face_data_;
    SpectrumType spectrum_;
    
    //! callAsync 呼び出し中に確保するべきミューテックス。
    /* AudioPluginAudioProcessorEditor デストラクト時に、
     * 画像処理スレッドを停止したあとで callAsync の処理が呼び出されてアクセス違反が発生することがある。
     * これを回避するため、　callAsync　の中でこのミューテックスを確保することとし、
     * このミューテックスが破棄されているときは何も処理をしないようにする。
     */
    std::shared_ptr<std::mutex> mtx_call_async_;

private:
    void setFaceData(FaceData const &fd)
    {
        std::unique_lock lock(gui_mtx_);
        face_data_ = fd;
    }
    
    double getMouthAspectRatio(std::vector<dlib::dpoint> const &points)
    {
        auto distance = [](dlib::dpoint pt1, dlib::dpoint pt2) {
            auto const x = pt2.x() - pt1.x();
            auto const y = pt2.y() - pt1.y();
            return std::sqrt(x * x + y * y);
        };
        
        auto const A = distance(points[2], points[10]);
        auto const B = distance(points[4], points[8]);
        auto const C = distance(points[0], points[6]);
        
        auto const mar = (A + B) / (2.0 * C);
        
        return mar;
    }
    
    struct ImageProcessingContext {
        ImageProcessingContext()
        {
            detector = dlib::get_frontal_face_detector();

            juce::File executable_dir = juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
            juce::File landmark_file = executable_dir.getChildFile("Data/shape_predictor_68_face_landmarks.dat");
            if(landmark_file.existsAsFile() == false) {
                landmark_file = executable_dir.getParentDirectory().getChildFile("Resources/shape_predictor_68_face_landmarks.dat");
            }

#if JUCE_WINDOWS
            std::ifstream ifs(landmark_file.getFullPathName().toWideCharPointer(), std::ios::binary | std::ios::in);
#else
            std::ifstream ifs(landmark_file.getFullPathName().toUTF8(), std::ios::binary | std::ios::in);
#endif

            dlib::deserialize(predictor, ifs);

            tmp_mouth_points.resize(kNumMouthPoints);
            set_image_size(tmp_array, kProcessImageHeight, kProcessImageWidth);
        }
        
        dlib::frontal_face_detector detector;
        dlib::shape_predictor predictor;

        std::vector<dlib::dpoint> tmp_mouth_points;
        dlib::array2d<unsigned char> tmp_array;
    };
    
    ImageProcessingContext context_;
    
    //! camera から取得した画像を、画像処理スレッドで処理するために保持しておく変数
    juce::Image tmp_image_;
    
    void ProcessImage(juce::Image img, ImageProcessingContext &ctx)
    {
        static bool write_image = false;
        auto get_desktop_file = [](juce::String const &filename) {
            return juce::File::getSpecialLocation(juce::File::userDesktopDirectory).getChildFile(filename);
        };
        
        if(write_image) {
            juce::PNGImageFormat fmt;
            juce::File file = get_desktop_file("original_image.png");
            auto fs = file.createOutputStream();
            fmt.writeImageToStream(img, *fs);
            fs.reset();
        }
        
        // expected image ascpected ratio
        double const ar = (double)kDefaultWidth / kDefaultHeight;
        
        auto const actual_w = img.getWidth();
        auto const expected_w = (int)std::round(img.getHeight() * ar);
        
        if(actual_w == expected_w) {
            // do nothing.
        } else if(actual_w > expected_w) {
            // 画面が横長なので左右を捨てる。
            int const nx = actual_w - expected_w;
            auto rect = img.getBounds().reduced(nx / 2.0, 0);
            img = img.getClippedImage(rect);
        } else {
            // 画面が縦長なので上下を捨てる。
             int const ny = img.getHeight() - (img.getWidth() / ar);
             auto rect = img.getBounds().reduced(0, ny / 2.0);
             img = img.getClippedImage(rect);
        }
        
        img = img.rescaled(kProcessImageWidth, kProcessImageHeight);
        img.desaturate();

        if(write_image) {
            juce::PNGImageFormat fmt;
            juce::File file = get_desktop_file("juce_image.png");
            auto fs = file.createOutputStream();
            fmt.writeImageToStream(img, *fs);
            fs.reset();
        }
        
        if(img.getPixelData()->pixelFormat == juce::Image::PixelFormat::RGB) {
            dlib::assign_image(ctx.tmp_array, JuceImageArray2d<dlib::rgb_pixel>(img));
        } else {
            dlib::assign_image(ctx.tmp_array, JuceImageArray2d<dlib::rgb_alpha_pixel>(img));
        }
        
#ifdef DLIB_PNG_SUPPORT
        if(write_image) {
            dlib::save_png(ctx.tmp_array, get_desktop_file("dlib_image.png").getFullPathName().toStdString());
        }
#endif
        
        auto rects = ctx.detector(ctx.tmp_array, 0);
        
        FaceData tmp_face_data = getFaceData();
                    
        if(rects.empty()) {
            tmp_face_data.num_skipped_ += 1;
            if(kShowFace) {
                tmp_face_data.image_ = std::move(img);
            }
            
            setFaceData(tmp_face_data);
        } else {
            dlib::full_object_detection shape = ctx.predictor(ctx.tmp_array, rects[0]);

            for(int i = 0, end = kNumMouthPoints; i < end; ++i) {
                ctx.tmp_mouth_points[i] = shape.part(i + kMouthPointsBegin);
            }
            
            double const tmp_mar = getMouthAspectRatio(ctx.tmp_mouth_points);
            
            for(int i = 0, end = kNumMouthPoints; i < end; ++i) {
                tmp_face_data.mouth_points_[i].Push(ctx.tmp_mouth_points[i]);
            }

            tmp_face_data.mar_.Push(tmp_mar);
            tmp_face_data.num_processed_ += 1;
            tmp_face_data.num_skipped_ = 0;
            
            if(kShowFace) {
                tmp_face_data.image_ = std::move(img);
            }
            
            setFaceData(tmp_face_data);
        }
    }
    
    void run() override
    {
        int continue_count_ = 0;
        
        for( ; threadShouldExit() == false; ) {
            
            juce::Image tmp_image = std::move(tmp_image_);
            
            if(tmp_image.isValid() == false) {
                continue_count_ += 1;
                if(continue_count_ < 10) {
                    // do nothing.
                } else if(continue_count_ < 30) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                
                continue;
            }
            
            continue_count_ = 0;
                        
            ProcessImage(tmp_image, context_);
            
            auto mm = juce::MessageManager::getInstance();
            mm->callAsync([this, w = std::weak_ptr<std::mutex>(mtx_call_async_)] {
                
                for( ; ; ) {
                    auto const pmtx = w.lock();
                    if(!pmtx) {
                        // callAsync 呼び出し中に確保するべき mutex が破棄されている。
                        // => 何も処理をせずにリターンする。
                        return;
                    }
                    
                    std::unique_lock lock(*pmtx, std::try_to_lock);
                    if(lock) {
                        owner_->OnUpdateMouthAspectRatio();
                        break;
                    }
                    
                    std::this_thread::yield();
                }
            });
        }
    }

    void imageReceived(juce::Image const &image) override
    {
#if WAHWTH_USE_IMAGE_PROCESSING_THREAD
        tmp_image_ = image;
#else
        ProcessImage(image, context_);
        
        auto mm = juce::MessageManager::getInstance();
        mm->callAsync([this] {
            owner_->OnUpdateMouthAspectRatio();
        });
#endif
    }
};

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
:   AudioProcessorEditor(&p)
,   processorRef(p)
,   pimpl_(std::make_unique<Impl>(this))
{
#if JUCE_WINDOWS
    String typeFaceName = "Meiryo UI";
    Desktop::getInstance().getDefaultLookAndFeel().setDefaultSansSerifTypefaceName(typeFaceName);
#elif JUCE_MAC
    String typeFaceName = "Arial Unicode MS";
    Desktop::getInstance().getDefaultLookAndFeel().setDefaultSansSerifTypefaceName(typeFaceName);
#endif
    
    juce::ignoreUnused (processorRef);
    
    setResizable(true, true);
    setResizeLimits(kMinWidth, kMinHeight, kMaxWidth, kMaxHeight);
    
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (kDefaultWidth, kDefaultHeight);
    
    addAndMakeVisible(pimpl_->cmb_camera_list_);
    addAndMakeVisible(pimpl_->sl_qfactor_);
    addAndMakeVisible(pimpl_->sl_low_freq_);
    addAndMakeVisible(pimpl_->sl_high_freq_);
    addAndMakeVisible(pimpl_->lbl_low_freq_);
    addAndMakeVisible(pimpl_->lbl_high_freq_);
    addAndMakeVisible(pimpl_->lbl_qfactor_);
    
    pimpl_->cmb_camera_list_.addItemList(juce::CameraDevice::getAvailableDevices(), 1);
    pimpl_->sl_qfactor_.setSliderStyle(juce::Slider::LinearBar);
    pimpl_->sl_low_freq_.setSliderStyle(juce::Slider::LinearBar);
    pimpl_->sl_high_freq_.setSliderStyle(juce::Slider::LinearBar);
    
    pimpl_->sl_low_freq_.setTextValueSuffix(" Hz");
    pimpl_->sl_high_freq_.setTextValueSuffix(" Hz");
    
    auto make_value_from_text_function = [](juce::Slider *slider) {
        return [slider](juce::String const &str) -> double {
            auto range = slider->getRange();
            try {
                return std::clamp<double>(std::stof(str.toStdString()), range.getStart(), range.getEnd());
            } catch(...) {
                return range.getStart();
            }
        };
    };
    
    auto make_text_from_value_function = [](juce::Slider *slider) {
        return [slider](double value) -> juce::String {
            char buf[32] = {};
            sprintf(buf, "%0.2f", value);
            return buf;
        };
    };
    
    pimpl_->sl_low_freq_.valueFromTextFunction = make_value_from_text_function(&pimpl_->sl_low_freq_);
    pimpl_->sl_low_freq_.textFromValueFunction = make_text_from_value_function(&pimpl_->sl_low_freq_);
    pimpl_->sl_high_freq_.valueFromTextFunction = make_value_from_text_function(&pimpl_->sl_high_freq_);
    pimpl_->sl_high_freq_.textFromValueFunction = make_text_from_value_function(&pimpl_->sl_high_freq_);
    pimpl_->sl_qfactor_.valueFromTextFunction = make_value_from_text_function(&pimpl_->sl_qfactor_);
    pimpl_->sl_qfactor_.textFromValueFunction = make_text_from_value_function(&pimpl_->sl_qfactor_);
    
    using AP = AudioPluginAudioProcessor;
    pimpl_->sl_low_freq_.setRange(AP::kLowFreqMin, AP::kLowFreqMax);
    pimpl_->sl_high_freq_.setRange(AP::kHighFreqMin, AP::kHighFreqMax);
    pimpl_->sl_qfactor_.setRange(AP::kQFactorMin, AP::kQFactorMax);
    
    pimpl_->sl_low_freq_.setValue(*processorRef.low_freq_, juce::dontSendNotification);
    pimpl_->sl_high_freq_.setValue(*processorRef.high_freq_, juce::dontSendNotification);
    pimpl_->sl_qfactor_.setValue(*processorRef.qfactor_, juce::dontSendNotification);
    
    pimpl_->sl_low_freq_.setSkewFactor(0.6);
    pimpl_->sl_high_freq_.setSkewFactor(0.6);
    pimpl_->sl_qfactor_.setSkewFactor(1.0);
    
    pimpl_->cmb_camera_list_.addListener(this);
    pimpl_->sl_low_freq_.addListener(this);
    pimpl_->sl_high_freq_.addListener(this);
    pimpl_->sl_qfactor_.addListener(this);
    
    pimpl_->lbl_low_freq_.setText ("Freq Lo", juce::NotificationType::dontSendNotification);
    pimpl_->lbl_high_freq_.setText("Freq Hi", juce::NotificationType::dontSendNotification);
    pimpl_->lbl_qfactor_.setText  ("Resonance", juce::NotificationType::dontSendNotification);
    
    auto const num_cameras = pimpl_->cmb_camera_list_.getNumItems();
    if(num_cameras != 0) {
        // 保存されたカメラ番号
        auto const try_first = std::clamp(processorRef.camera_index_.load(),
                                          AP::kCameraIndexMin,
                                          num_cameras);

        // カメラのオープンを試す順序を用意する。
        // 保存されたカメラ番号を最初に試す。
        // 一番最初にオープンできたカメラを使用する。
        std::vector<int> camera_open_order;
        camera_open_order.push_back(try_first);
        for(int i = 0, end = num_cameras; i < end; ++i) {
            if(i != try_first) { camera_open_order.push_back(i); }
        }

        for(auto index : camera_open_order) {
            if(pimpl_->openCamera(index)) {
                pimpl_->cmb_camera_list_.setSelectedItemIndex(index,
                                                              juce::NotificationType::dontSendNotification
                                                              );
                break;
            }
        }
    }

    processorRef.on_load_camera_index_ = [this](int index) {
        index = std::clamp<int>(index,
                                0,
                                pimpl_->cmb_camera_list_.getNumItems());
        
        pimpl_->cmb_camera_list_.setSelectedItemIndex(index);
    };
    
    getConstrainer()->setFixedAspectRatio((double)kMaxWidth / kMaxHeight);
    setBounds(0, 0, kDefaultWidth, kDefaultHeight);
    resized();
    
#if WAHWTH_USE_IMAGE_PROCESSING_THREAD
    pimpl_->startImageProcessingThread();
#endif
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    processorRef.on_load_camera_index_ = nullptr;
    
    pimpl_->closeCamera();
    
#if WAHWTH_USE_IMAGE_PROCESSING_THREAD
    pimpl_->stopImageProcessingThread();
#endif
    
    pimpl_.reset();
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
    auto const face_data = pimpl_->getFaceData();
    
    if(kShowFace && face_data.image_.isValid()) {
        int const kThumbnailWidth = 120;
        int const kThumbnailX = 10;
        int const kThumbnailY = 10;
        
        double scale = face_data.image_.getWidth() / (double)kThumbnailWidth;
        juce::AffineTransform t;
        t = t.scaled(-1, 1.0); // カメラから取得した画像を、顔の動きに合うように左右反転して表示する。
        t = t.scaled(1.0 / scale);
        t = t.translated(kThumbnailWidth + kThumbnailX, kThumbnailY);
        g.drawImageTransformed(face_data.image_, t);
     }
    
    g.setFont (15.0f);
    
    auto &points = face_data.mouth_points_;
    
    auto const col_bright = juce::Colour(0.3f, 0.7f, 0.9f, 0.75f);
    auto const col_dark = juce::Colour(0.3f, 0.8f, 0.2f, 0.75f);

    g.setColour(col_bright);
    
    // 口の形状を描画する
    if(face_data.num_skipped_ <= 2) {
        int left = 10000, top = 10000, right = 0, bottom = 0;
        
        for(auto const &pts: points) {
            auto avg = pts.GetAverage();
            left = std::min<int>(left, avg.x());
            right = std::max<int>(right, avg.x());
            top = std::min<int>(top, avg.y());
            bottom = std::max<int>(bottom, avg.y());
        }
        
        juce::Rectangle<int> b_mouth {
            left,
            top,
            std::max<int>(1, (right - left)), // 0除算を回避
            std::max<int>(1, (bottom - top))  // 0除算を回避
        };
        
        juce::Rectangle<int> b_area = getBounds();
        
        b_area = b_area.reduced((int)(b_area.getWidth() * 0.2), (int)(b_area.getHeight() * 0.2));
        
        auto const shift_mouth = b_mouth.getCentre();
        auto const shift_area = b_area.getCentre();
        
        double const scale_x = b_area.getWidth() / (double)b_mouth.getWidth();
        double const scale_y = b_area.getHeight() / (double)b_mouth.getHeight();
        double const scale = std::min<double>(scale_x, scale_y);
        
        b_mouth = b_mouth.expanded(b_mouth.getWidth() * (scale - 1.0) / 2.0,
                                   b_mouth.getHeight() * (scale - 1.0) / 2.0);
        
        auto drawLine = [&](int a, int b) {
            auto avg_a = points[a].GetAverage();
            auto avg_b = points[b].GetAverage();
            
            auto ax = avg_a.x();
            auto ay = avg_a.y();
            auto bx = avg_b.x();
            auto by = avg_b.y();
            
            ax = (ax - b_mouth.getCentre().x) * -1 + b_mouth.getCentre().x;
            bx = (bx - b_mouth.getCentre().x) * -1 + b_mouth.getCentre().x;
            
            g.drawLine((ax - shift_mouth.x) * scale + shift_area.x,
                       (ay - shift_mouth.y) * scale + shift_area.y,
                       (bx - shift_mouth.x) * scale + shift_area.x,
                       (by - shift_mouth.y) * scale + shift_area.y
                       );
        };
        
        for(int i = 0; i < 12; ++i) {
            drawLine(i, (i+1)%12);
        }
        
        for(int i = 0; i < 8; ++i) {
            drawLine(i+12, ((i+1)%8)+12);
        }
    }
    
    // スペクトルを描画する
    auto const &spectrum = pimpl_->calculateLogAmplitudeSpectrum();
    auto b = getBounds().reduced(10);
    
    b.removeFromTop(90);
    
    g.setColour(col_bright);
    
    static double kLogEmphasisCoeff = 200; // この値が大きいほうが低音域が広く表示される。 [9 .. inf]
    float last_x = -1;
    for(int i = 0, end = spectrum.size()-1; i < end; ++i) {
        auto &s1 = spectrum[i];
        auto &s2 = spectrum[i+1];
        
        float x1 = log10((i / (double)end) * kLogEmphasisCoeff + 1.0) / log10(kLogEmphasisCoeff + 1.0) * b.getWidth() + b.getX();
        float x2 = log10(((i+1) / (double)end) * kLogEmphasisCoeff + 1.0) / log10(kLogEmphasisCoeff + 1.0) * b.getWidth() + b.getX();
        float y1 = (-s1 / (double)120) * b.getHeight() + b.getY();
        float y2 = (-s2 / (double)120) * b.getHeight() + b.getY();

        assert(last_x < x1);
        assert(x1 < x2);
        g.drawLine(x1, y1, x2, y2, 2.0);
    }
    
    char const nl = '\n';
    std::stringstream ss;
    
    ss
    << "Mouth Aspect Ratio : " << face_data.mar_.GetAverage();
    
    g.drawFittedText (ss.str(), getLocalBounds().removeFromTop(100), juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    
    auto b = getBounds().reduced(10);
    auto right = b.removeFromRight(b.getWidth() / 3);
    pimpl_->cmb_camera_list_.setBounds(right.removeFromTop(20));
    right.removeFromTop(2);
    
    auto align_label_and_control = [](juce::Component &label,
                                      juce::Component &control,
                                      juce::Rectangle<int> b) {
        label.setBounds(b.removeFromLeft((int)(b.getWidth() * 0.33)));
        control.setBounds(b);
    };
        
    align_label_and_control(pimpl_->lbl_low_freq_, pimpl_->sl_low_freq_, right.removeFromTop(20));
    right.removeFromTop(2);
    
    align_label_and_control(pimpl_->lbl_high_freq_, pimpl_->sl_high_freq_, right.removeFromTop(20));
    right.removeFromTop(2);
    
    align_label_and_control(pimpl_->lbl_qfactor_, pimpl_->sl_qfactor_, right.removeFromTop(20));
}

void AudioPluginAudioProcessorEditor::OnUpdateMouthAspectRatio()
{
    auto const face_data = pimpl_->getFaceData();
    
    double const kMarMin = 0.4;
    double const kMarMax = 0.95;
    
    double const mar = std::clamp(face_data.mar_.GetAverage(), kMarMin, kMarMax);
    double const freq = (mar - kMarMin) / (kMarMax - kMarMin);
    
    auto *param = processorRef.freq_;
    param->setValueNotifyingHost(freq);

    repaint();
}

void AudioPluginAudioProcessorEditor::sliderValueChanged(juce::Slider *s)
{
    if(s == &pimpl_->sl_qfactor_) {
        auto *param = processorRef.qfactor_;
        param->setValueNotifyingHost(param->convertTo0to1(s->getValue()));
    } else if(s == &pimpl_->sl_low_freq_) {
        auto *param = processorRef.low_freq_;
        param->setValueNotifyingHost(param->convertTo0to1(s->getValue()));
    } else if(s == &pimpl_->sl_high_freq_) {
        auto *param = processorRef.high_freq_;
        param->setValueNotifyingHost(param->convertTo0to1(s->getValue()));
    } else {
        assert(false);
    }
}

void AudioPluginAudioProcessorEditor::comboBoxChanged(juce::ComboBox *c)
{
    if(c == &pimpl_->cmb_camera_list_) {
        auto const item_index = c->getSelectedItemIndex();
        if(pimpl_->openCamera(item_index)) {
            processorRef.camera_index_ = item_index;
        } else {
            c->setSelectedId(0, juce::NotificationType::dontSendNotification);
        }
    } else {
        assert(false);
    }
}

void AudioPluginAudioProcessorEditor::mouseUp(juce::MouseEvent const&e)
{
    juce::PopupMenu menu;
    menu.addItem(1, "About...");
    
    int result = menu.show();
    if(result == 1) {
        showModalAboutDialog(this);
    }
}
