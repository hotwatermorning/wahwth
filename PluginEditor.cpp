#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <algorithm>
#include <opencv4/opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

bool kShowFace = true;

int const kMinWidth = 400;
int const kMinHeight = 300;
int const kMaxWidth = 1200;
int const kMaxHeight = 900;
int const kDefaultWidth = 600;
int const kDefaultHeight = 450;

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
        juce::Thread::stopThread(1000);
    }
    
    struct FaceData
    {
        using MouthPointArray = std::array<MovingAverageValue<dlib::point>, kNumMouthPoints>;
        juce::Image image_;             //!< エディター画面で表示するカメラ画像
        MouthPointArray mouth_points_;  //!< エディター画面で表示する口の輪郭
        MovingAverageValue<double> mar_;
        int num_skipped_ = 0;
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
        
        cam_.reset(juce::CameraDevice::openDevice(index, kMinWidth, kMinHeight, kMaxWidth, kMaxHeight));
        if(cam_) {
            cam_->addListener(this);
            
            // 小ウィンドウに登録しない。
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
            cam_->removeListener(this);
            delete viewer_;
            viewer_ = nullptr;
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
    juce::ToggleButton tgl_bypass_;

    juce::Label lbl_bypass_;
    juce::Label lbl_low_freq_;
    juce::Label lbl_high_freq_;
    juce::Label lbl_qfactor_;
    
private:
    std::mutex mutable camera_mtx_; //!< カメラスレッドと画像処理スレッドの排他制御
    std::mutex mutable gui_mtx_;    //!< GUI スレッドと画像処理スレッドの排他制御
    juce::Image tmp_image_;         //!< camera から取得した画像を、画像処理スレッドで処理するために保持しておく変数
        
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
    
    double getMouthAspectRatio(std::vector<dlib::point> const &points)
    {
        auto distance = [](dlib::point pt1, dlib::point pt2) {
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
    
    void run() override
    {
        auto detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor predictor;

        // dlib の deserialize に std::stringstream を渡してもうまく読み込めなかったので、
        // 一時ファイルを作成してそのファイルから読み込むようにする。
        auto tmp_file = juce::File::createTempFile("shape_predictor");
        auto os = tmp_file.createOutputStream();
        os->write(BinaryData::shape_predictor_68_face_landmarks_dat,
                  BinaryData::shape_predictor_68_face_landmarks_datSize);

        dlib::deserialize(tmp_file.getFullPathName().toStdString()) >> predictor;
        
        std::vector<dlib::point> tmp_mouth_points(kNumMouthPoints);
        cv::Mat tmp_mat;
        
        for( ; threadShouldExit() == false; ) {
            
            juce::Image tmp_image;
                       
            {
                std::unique_lock lock(camera_mtx_);
                tmp_image = std::move(tmp_image_);
            }
            
            if(tmp_image.isValid() == false) {
                sleep(10);
                continue;
            }
            
            static bool write_image_ = false;
            
            if(write_image_) {
                juce::PNGImageFormat fmt;
                juce::File file("/Users/yuasa/Desktop/juce_image_original.png");
                auto fs = file.createOutputStream();
                fmt.writeImageToStream(tmp_image, *fs);
                fs.reset();
            }
            
            // expected image ascpected ratio
            double const ar = (double)kDefaultWidth / kDefaultHeight;
            
            auto const actual_w = tmp_image.getWidth();
            auto const expected_w = (int)std::round(tmp_image.getHeight() * ar);
            
            auto const DW = kDefaultWidth;
            auto const DH = kDefaultHeight;
            
            if(actual_w == expected_w) {
                // do nothing.
            } else if(actual_w > expected_w) {
                // 画面が横長なので左右を捨てる。
                int const nx = actual_w - expected_w;
                auto rect = tmp_image.getBounds().reduced(nx / 2.0, 0);
                tmp_image = tmp_image.getClippedImage(rect);
            } else {
                // 画面が縦長なので上下を捨てる。
                 int const ny = tmp_image.getHeight() - (tmp_image.getWidth() / ar);
                 auto rect = tmp_image.getBounds().reduced(0, ny / 2.0);
                 tmp_image = tmp_image.getClippedImage(rect);
            }
             
            auto rescaled = tmp_image.rescaled(kDefaultWidth, kDefaultHeight);

            if(write_image_) {
                juce::PNGImageFormat fmt;
                juce::File file("/Users/yuasa/Desktop/juce_image.png");
                auto fs = file.createOutputStream();
                fmt.writeImageToStream(rescaled, *fs);
                fs.reset();
            }
            
            juce::Image::BitmapData bmp_rescaled(rescaled, juce::Image::BitmapData::readWrite);
            
            cv::Mat mat_rescaled;
            
            if(rescaled.getPixelData()->pixelFormat == juce::Image::PixelFormat::RGB) {
                mat_rescaled = cv::Mat(DH, DW, CV_8UC3, bmp_rescaled.data);
            } else if(rescaled.getPixelData()->pixelFormat == juce::Image::PixelFormat::ARGB) {
                mat_rescaled = cv::Mat(DH, DW, CV_8UC4, bmp_rescaled.data);
            } else {
                return;
            }
            
            
            if(write_image_) {
                cv::imwrite("/Users/yuasa/Desktop/cv_rescaled.png", mat_rescaled);
            }
            
            cv::cvtColor(mat_rescaled,
                         tmp_mat,
                         cv::COLOR_RGBA2GRAY
                         );
            
            auto img = dlib::cv_image<unsigned char>(tmp_mat);
            
            auto rects = detector(img, 0);
            
            FaceData tmp_face_data = getFaceData();
            
            if(kShowFace) {
                cv::cvtColor(tmp_mat,
                             mat_rescaled,
                             cv::COLOR_GRAY2RGBA
                             );
            }
            
            if(rects.empty()) {
                tmp_face_data.num_skipped_ += 1;
                if(kShowFace) {
                    tmp_face_data.image_ = std::move(rescaled);
                }
                
                setFaceData(tmp_face_data);
            } else {
                dlib::full_object_detection shape = predictor(img, rects[0]);

                for(int i = 0, end = kNumMouthPoints; i < end; ++i) {
                    tmp_mouth_points[i] = shape.part(i + kMouthPointsBegin);
                }
                
                double const tmp_mar = getMouthAspectRatio(tmp_mouth_points);
                
                for(int i = 0, end = kNumMouthPoints; i < end; ++i) {
                    tmp_face_data.mouth_points_[i].Push(tmp_mouth_points[i]);
                }

                tmp_face_data.mar_.Push(tmp_mar);
                tmp_face_data.num_processed_ += 1;
                
                if(kShowFace) {
                    tmp_face_data.image_ = std::move(rescaled);
                }
                
                setFaceData(tmp_face_data);
            }
            
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
        std::unique_lock lock(camera_mtx_);
        tmp_image_ = image;
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
    addAndMakeVisible(pimpl_->tgl_bypass_);
    addAndMakeVisible(pimpl_->sl_qfactor_);
    addAndMakeVisible(pimpl_->sl_low_freq_);
    addAndMakeVisible(pimpl_->sl_high_freq_);
    addAndMakeVisible(pimpl_->lbl_bypass_);
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
    
    pimpl_->sl_low_freq_.setValue(AP::kLowFreqDefault, juce::dontSendNotification);
    pimpl_->sl_high_freq_.setValue(AP::kHighFreqDefault, juce::dontSendNotification);
    pimpl_->sl_qfactor_.setValue(AP::kQFactorDefault, juce::dontSendNotification);
    
    pimpl_->sl_low_freq_.setSkewFactor(0.6);
    pimpl_->sl_high_freq_.setSkewFactor(0.6);
    pimpl_->sl_qfactor_.setSkewFactor(1.0);
    
    pimpl_->cmb_camera_list_.addListener(this);
    pimpl_->tgl_bypass_.addListener(this);
    pimpl_->sl_low_freq_.addListener(this);
    pimpl_->sl_high_freq_.addListener(this);
    pimpl_->sl_qfactor_.addListener(this);
    
    pimpl_->lbl_bypass_.setText   ("Bypass", juce::NotificationType::dontSendNotification);
    pimpl_->lbl_low_freq_.setText ("Freq Lo", juce::NotificationType::dontSendNotification);
    pimpl_->lbl_high_freq_.setText("Freq Hi", juce::NotificationType::dontSendNotification);
    pimpl_->lbl_qfactor_.setText  ("Q Factor", juce::NotificationType::dontSendNotification);
    
    for(int i = 0, end = pimpl_->cmb_camera_list_.getNumItems(); i < end; ++i) {
        if(pimpl_->openCamera(i)) {
            pimpl_->cmb_camera_list_.setSelectedItemIndex(i, juce::NotificationType::dontSendNotification);
            break;
        }
    }
    
    getConstrainer()->setFixedAspectRatio((double)kMaxWidth / kMaxHeight);
    setBounds(0, 0, kDefaultWidth, kDefaultHeight);
    resized();
    
    pimpl_->startImageProcessingThread();
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    pimpl_->closeCamera();
    pimpl_->stopImageProcessingThread();
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
    
    g.setColour(juce::Colour(0.3f, 0.7f, 0.9f, 0.75f));
    
    // 口の形状を描画する
    if(points.size() > 0) {
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
    auto const b = getBounds().reduced(30);
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
    pimpl_->cmb_camera_list_.setBounds(right.removeFromTop(16));
    right.removeFromTop(2);
    
    auto align_label_and_control = [](juce::Component &label,
                                      juce::Component &control,
                                      juce::Rectangle<int> b) {
        label.setBounds(b.removeFromLeft((int)(b.getWidth() * 0.33)));
        control.setBounds(b);
    };
    
    align_label_and_control(pimpl_->lbl_bypass_, pimpl_->tgl_bypass_, right.removeFromTop(16));
    right.removeFromTop(2);
    
    align_label_and_control(pimpl_->lbl_low_freq_, pimpl_->sl_low_freq_, right.removeFromTop(16));
    right.removeFromTop(2);
    
    align_label_and_control(pimpl_->lbl_high_freq_, pimpl_->sl_high_freq_, right.removeFromTop(16));
    right.removeFromTop(2);
    
    align_label_and_control(pimpl_->lbl_qfactor_, pimpl_->sl_qfactor_, right.removeFromTop(16));
    right.removeFromTop(2);
}

void AudioPluginAudioProcessorEditor::OnUpdateMouthAspectRatio()
{
    auto const face_data = pimpl_->getFaceData();
    
    double const kMarMin = 0.4;
    double const kMarMax = 0.95;
    
    double const mar = std::clamp(face_data.mar_.GetAverage(), kMarMin, kMarMax);
    double const freq = (mar - kMarMin) / (kMarMax - kMarMin);
    
    auto *param = processor.getParameters()[AudioPluginAudioProcessor::Frequency];
    param->setValueNotifyingHost(freq);

    repaint();
}

void AudioPluginAudioProcessorEditor::buttonStateChanged(juce::Button *b)
{
    assert(b == &pimpl_->tgl_bypass_);
    
    if(b == &pimpl_->tgl_bypass_) {
        auto *param = processorRef.bypass_;
        param->setValueNotifyingHost(param->convertTo0to1(b->getToggleState()));
    } else {
        assert(false);
    }
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
            // do nothing.
        } else {
            c->setSelectedId(0, juce::NotificationType::dontSendNotification);
        }
    } else {
        assert(false);
    }
}
